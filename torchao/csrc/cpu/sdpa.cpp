#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <ATen/Tensor.h>
#include <limits>
#include <omp.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/native/CPUBlas.h>

namespace torchao {

namespace {

template <typename T>
struct is_reduced_floating_point:
    std::integral_constant<bool,
      std::is_same_v<T, at::Half> ||
      std::is_same_v<T, at::BFloat16>> {
};

template <typename T>
constexpr bool is_reduced_floating_point_v = is_reduced_floating_point<T>::value;

inline double calculate_scale(
    const at::Tensor& query,
    double scale) {
  return scale == 0.0 ? 1.0 / std::sqrt(query.size(-1)) : scale;
}

// out = val * a + b
// is_b_stride_zero: If the stride of b is 0 (mask broadcasting case),
//                take b as a scalar pointer.
template <bool is_b_stride_zero, typename T1, typename T2>
inline void _scale_attn_mask_fusion_kernel(
    T1* a,
    T2* b,
    const int& size,
    T1* out,
    T1& val) {
  const auto vec_size1 = at::vec::Vectorized<T1>::size();
  const auto vec_size2 = at::vec::Vectorized<T2>::size();
  constexpr int64_t T1_n =
      (vec_size2 == vec_size1 * 2 && is_reduced_floating_point_v<T2>) ? 2 : 1;
  constexpr int64_t T2_n = 1;
  auto vec_scale = at::vec::VectorizedN<T1, T1_n>(val);
  int64_t i = 0;
  for (; i < size - (size % vec_size2); i += vec_size2) {
    auto a_n = at::vec::VectorizedN<T1, T1_n>::loadu(a + i);
    at::vec::VectorizedN<T2, T2_n> b_n;
    if constexpr(is_b_stride_zero) {
      b_n = at::vec::VectorizedN<T2, T2_n>((T1)b[0]);
    } else {
      b_n = at::vec::VectorizedN<T2, T2_n>::loadu(b + i);
    }
    auto b_n_convert = at::vec::convert<T1, T1_n, T2, T2_n, true>(b_n);
    auto res = a_n * vec_scale + b_n_convert;
    res.store(out + i);
  }
  for (; i < size; i++) {
    auto tmp0 = a[i];
    T1 tmp1;
    if constexpr(is_b_stride_zero) {
      tmp1 = (T1)b[0];
    } else {
      tmp1 = (T1)b[i];
    }
    out[i] = tmp0 * val + tmp1;
  }
}

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1>& x, at::vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    _store(out + i, tmp1);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(tmp_max, vec_tmp_max.reduce_max());
}

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <typename scalar_t,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

void reshape_attn_mask_to_4d(
    at::Tensor& attn_mask,
    int64_t batchSize,
    int64_t num_head,
    int64_t qSize,
    int64_t kvSize) {
  // Support mask shapes:
  // 2d: ({Q_seq_len, 1}  x {KV_seq_len, 1})
  // 4d: ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})
  // Guaranteed in check_attn_mask_shape
  int64_t attn_mask_size_0 = 1;
  int64_t attn_mask_size_1 = 1;
  if (attn_mask.dim() == 4) {
    if (attn_mask.size(0) == batchSize) {
      attn_mask_size_0 = batchSize;
    }
    if (attn_mask.size(1) == num_head) {
      attn_mask_size_1 = num_head;
    }
  }
  attn_mask = attn_mask
                .view({attn_mask_size_0, attn_mask_size_1, attn_mask.size(-2), attn_mask.size(-1)})
                .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
}

// TODO: Use at::native::_store instead when it supports Half.
template <typename scalar_t>
inline void _store(scalar_t* dst, at::vec::Vectorized<scalar_t> src, int size=at::vec::Vectorized<scalar_t>::size()) {
  src.store(dst, size);
}

template <typename scalar_t>
inline typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, void>
_store(scalar_t* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert_from_float<scalar_t>(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

template <typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char> || std::is_same_v<scalar_t, signed char>, void>
_store(scalar_t* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert<scalar_t>(src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

template <typename scalar_t>
inline void pad_row_zero(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  int i = 0;
  for (; i < rows - 1; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - j);
    }
  }

  // zero padding
  int j = 0;
  for (; j < cols - (cols % vec_size); j += vec_size) {
    auto vec_v = at::vec::Vectorized<scalar_t>(0);
    vec_v.store(padding_value_ptr + i * cols + j);
  }

  if (j < cols) {
    auto vec_v = at::vec::Vectorized<scalar_t>(0);
    vec_v.store(padding_value_ptr + i * cols + j, cols - j);
  }
}

template <typename scalar_t>
inline void pad_row_128_padding(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi,
    int padding) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  int i = 0;
  for (; i < rows - padding; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - j);
    }
  }

  // 128 padding
  for (; i < rows; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v = at::vec::Vectorized<scalar_t>(128);
      vec_v.store(padding_value_ptr + i * cols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>(128);
      vec_v.store(padding_value_ptr + i * cols + j, cols - j);
    }
  }
}

template <typename scalar_t>
inline void pad_col_zero(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  for (int i = 0; i < rows; i++) {
    int j = 0;
    for (; j < cols - 1 - ((cols - 1) % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }
    if (j < cols - 1) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - 1 - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - 1 - j);
      *(padding_value_ptr + i * cols + cols - 1) = scalar_t(0);
    }
  }
}

template <typename scalar_t>
inline void pad_col_zero_padding(
    scalar_t* value_ptr,
    scalar_t* padding_value_ptr,
    int rows,
    int cols,
    int ldi,
    int padding) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  for (int i = 0; i < rows; i++) {
    int j = 0;
    for (; j < cols - padding - ((cols - padding) % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(padding_value_ptr + i * cols + j);
    }
    if (j < cols - padding) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - padding - j);
      vec_v.store(padding_value_ptr + i * cols + j, cols - padding - j);
      *(padding_value_ptr + i * cols + cols - padding) = scalar_t(0);
    }
  }
}

/*
1. dequant
2. add mask
3. max reduce for softmax
*/
template <typename mask_t>
inline void _dequant_mask_max_fusion_kernel(
    const int32_t* in,
    const mask_t* mask_ptr,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldm, // leading dimension mask
    const int& ldo,
    const int32_t& beta, // zp_a*zp_b*k
    const float& alpha, // scale_a*scale_b*scale_sdpa
    float* out,
    float* sfm_max_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  auto vec_beta = at::vec::Vectorized<int32_t>(beta);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    float* tmp_out = out + row * ldo;
    const mask_t* mask_data_ptr = mask_ptr + row * ldm;
    float tmp_max = -std::numeric_limits<float>::infinity();
    auto vec_tmp_max = at::vec::Vectorized<float>(tmp_max);
    for (long col = 0; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = at::vec::Vectorized<mask_t>::loadu(mask_data_ptr + col);
      auto tmp7 = at::vec::convert<float>(tmp6);
      auto tmp8 = tmp5 + tmp7;
      vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp8);
      _store(tmp_out + col, tmp8);
    }
    tmp_max = std::max(tmp_max, vec_tmp_max.reduce_max());
    for (long col = vec_size * (N / vec_size); col < N; col++) {
      auto sum_b = sum_b_ptr[col];
      auto tmp0 = tmp_in[col];
      auto tmp1 = tmp0 - sum_b;
      auto tmp2 = tmp1 - sum_a;
      auto tmp3 = tmp2 + beta;
      auto tmp4 = (float) tmp3;
      auto tmp5 = tmp4 * alpha;
      auto tmp6 = mask_data_ptr[col];
      auto tmp7 = (float) tmp6;
      auto tmp8 = tmp5 + tmp7;
      tmp_max = std::max(tmp_max, tmp8);
      tmp_out[col] = tmp8;
    }
    sfm_max_ptr[row] = std::max(sfm_max_ptr[row], tmp_max);
  }
}

/*
1. dequant
2. max reduce for softmax
*/
inline void _dequant_max_fusion_kernel(
    const int32_t* in,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldo,
    const int32_t& beta, // zp_a*zp_b*k
    const float& alpha, // scale_a*scale_b*scale_sdpa
    float* out,
    float* sfm_max_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  auto vec_beta = at::vec::Vectorized<int32_t>(beta);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    float* tmp_out = out + row * ldo;
    float tmp_max = -std::numeric_limits<float>::infinity();
    auto vec_tmp_max = at::vec::Vectorized<float>(tmp_max);
    for (long col = 0; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp5);
      _store(tmp_out + col, tmp5);
    }
    tmp_max = std::max(tmp_max, vec_tmp_max.reduce_max());
    for (long col = vec_size * (N / vec_size); col < N; col++) {
      auto sum_b = sum_b_ptr[col];
      auto tmp0 = tmp_in[col];
      auto tmp1 = tmp0 - sum_b;
      auto tmp2 = tmp1 - sum_a;
      auto tmp3 = tmp2 + beta;
      auto tmp4 = (float) tmp3;
      auto tmp5 = tmp4 * alpha;
      tmp_max = std::max(tmp_max, tmp5);
      tmp_out[col] = tmp5;
    }
    sfm_max_ptr[row] = std::max(sfm_max_ptr[row], tmp_max);
  }
}

/*
1. Softmax: sub max, exp, sum reduce, div sum
2. quant
3. sum for attention
*/
template <typename scalar_t>
inline void _sub_exp_sum_div_quant_sum_fusion_kernel(
    const float* in,
    const int64_t& M,
    const int64_t& N_step,
    const int64_t& NSlice,
    const int& ldi,
    const int& ldo,
    const int& kvSize,
    const int& rndkvSplitSize,
    const int& av_gemm_K,
    const int32_t& beta1, // zp_a
    const int32_t& beta2, // zp_b
    const float& alpha, // scale_a
    float* local,
    scalar_t* out,
    float* sfm_max_ptr,
    float* sfm_sum_ptr,
    int32_t* sum_a_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  scalar_t zero = 0;
  auto vec_zero = at::vec::Vectorized<scalar_t>(zero);
  float beta1_float = (float) beta1;
  auto vec_beta1 = at::vec::Vectorized<float>(beta1_float);
  for (int64_t row = 0; row < M; ++row) {
    auto sfm_max = sfm_max_ptr[row];
    auto vec_max = at::vec::Vectorized<float>(sfm_max);
    // sub max, exp, sum reduce
    const float* qk_block_data = in + row * rndkvSplitSize;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      const float* tmp_in = qk_block_data + l * ldi;
      float tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<float>(tmp_sum);
      float* tmp_out = local + n;
      for (long col = 0; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        vec_tmp_sum += tmp2;
        _store(tmp_out + col, tmp2);
      }
      tmp_sum += vec_tmp_sum.reduce_add();
      for (long col = vec_size * (kvBlockSize / vec_size); col < kvBlockSize; col++) {
        auto tmp0 = tmp_in[col];
        auto tmp1 = tmp0 - sfm_max;
        auto tmp2 = exp(tmp1);
        tmp_sum += tmp2;
        tmp_out[col] = tmp2;
      }
      sfm_sum_ptr[row] += tmp_sum;
    }
    // div sum, sum for attention
    auto sum_scale = 1 / sfm_sum_ptr[row] / alpha;
    auto vec_sum_scale = at::vec::Vectorized<float>(sum_scale);
    scalar_t* qk_reduced_block_data = out + row * av_gemm_K;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      int32_t tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<int32_t>(tmp_sum);
      float* tmp_in = local + n;
      scalar_t* tmp_out = qk_reduced_block_data + l * ldo;
      for (long col = 0; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::maximum(tmp3, vec_min_val);
        auto tmp5 = at::vec::minimum(tmp4, vec_max_val);
        _store(tmp_out + col, tmp5);
        auto tmp6 = at::vec::convert<int32_t>(tmp5);
        vec_tmp_sum += tmp6;
      }
      tmp_sum += vec_tmp_sum.reduce_add();
      for (long col = vec_size * (kvBlockSize / vec_size); col < kvBlockSize; col++) {
        auto tmp0 = tmp_in[col];
        auto tmp1 = tmp0 * sum_scale;
        auto tmp2 = std::nearbyint(tmp1);
        auto tmp3 = tmp2 + beta1_float;
        auto tmp4 = std::max(tmp3, min_val);
        auto tmp5 = std::min(tmp4, max_val);
        tmp_out[col] = tmp5;
        auto tmp6 = (int32_t) tmp5;
        tmp_sum += tmp6;
      }
      sum_a_ptr[row] += tmp_sum * beta2;
      // set zero
      for (long col = kvBlockSize; col <  vec_size * (av_gemm_K / vec_size); col += vec_size) {
        _store(tmp_out + col, vec_zero);
      }
      for (long col = vec_size * (av_gemm_K / vec_size); col < av_gemm_K; col++) {
        tmp_out[col] = zero;
      }
    }
  }
}

template <typename scalar_t>
inline void _sub_exp_sum_div_quant_fusion_kernel(
    const float* in,
    const int64_t& M,
    const int64_t& N_step,
    const int64_t& NSlice,
    const int& ldi,
    const int& ldo,
    const int& kvSize,
    const int& rndkvSplitSize,
    const int& av_gemm_K,
    const int32_t& beta1, // zp_a
    const float& alpha, // scale_a
    float* local,
    scalar_t* out,
    float* sfm_max_ptr,
    float* sfm_sum_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  scalar_t zero = 0;
  auto vec_zero = at::vec::Vectorized<scalar_t>(zero);
  float beta1_float = (float) beta1;
  auto vec_beta1 = at::vec::Vectorized<float>(beta1_float);
  for (int64_t row = 0; row < M; ++row) {
    auto sfm_max = sfm_max_ptr[row];
    auto vec_max = at::vec::Vectorized<float>(sfm_max);
    // sub max, exp, sum reduce
    const float* qk_block_data = in + row * rndkvSplitSize;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      const float* tmp_in = qk_block_data + l * ldi;
      float tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<float>(tmp_sum);
      float* tmp_out = local + n;
      for (long col = 0; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        vec_tmp_sum += tmp2;
        _store(tmp_out + col, tmp2);
      }
      tmp_sum += vec_tmp_sum.reduce_add();
      for (long col = vec_size * (kvBlockSize / vec_size); col < kvBlockSize; col++) {
        auto tmp0 = tmp_in[col];
        auto tmp1 = tmp0 - sfm_max;
        auto tmp2 = exp(tmp1);
        tmp_sum += tmp2;
        tmp_out[col] = tmp2;
      }
      sfm_sum_ptr[row] += tmp_sum;
    }
    // div sum, sum for attention
    auto sum_scale = 1 / sfm_sum_ptr[row] / alpha;
    auto vec_sum_scale = at::vec::Vectorized<float>(sum_scale);
    scalar_t* qk_reduced_block_data = out + row * av_gemm_K;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      float* tmp_in = local + n;
      scalar_t* tmp_out = qk_reduced_block_data + l * ldo;
      for (long col = 0; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::maximum(tmp3, vec_min_val);
        auto tmp5 = at::vec::minimum(tmp4, vec_max_val);
        _store(tmp_out + col, tmp5);
      }
      for (long col = vec_size * (kvBlockSize / vec_size); col < kvBlockSize; col++) {
        auto tmp0 = tmp_in[col];
        auto tmp1 = tmp0 * sum_scale;
        auto tmp2 = std::nearbyint(tmp1);
        auto tmp3 = tmp2 + beta1_float;
        auto tmp4 = std::max(tmp3, min_val);
        auto tmp5 = std::min(tmp4, max_val);
        tmp_out[col] = tmp5;
      }
      // set zero
      for (long col = kvBlockSize; col <  vec_size * (av_gemm_K / vec_size); col += vec_size) {
        _store(tmp_out + col, vec_zero);
      }
      for (long col = vec_size * (av_gemm_K / vec_size); col < av_gemm_K; col++) {
        tmp_out[col] = zero;
      }
    }
  }
}

/*
1. dequant
2. quant
*/
template <typename scalar_t>
inline void _dequant_quant_fusion_kernel(
    const int32_t* in,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldo,
    const int32_t& beta1, // zp_a*zp_b*k
    const int32_t& beta2, // zp_c
    const float& alpha, // scale_a*scale_b/scale_c
    scalar_t* out) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  auto vec_beta1 = at::vec::Vectorized<int32_t>(beta1);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  float beta2_float = (float) beta2;
  auto vec_beta2 = at::vec::Vectorized<float>(beta2_float);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    scalar_t* tmp_out = out + row * ldo;
    for (long col = 0; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta1;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = tmp5.round();
      auto tmp7 = tmp6 + vec_beta2;
      auto tmp8 = at::vec::maximum(tmp7, vec_min_val);
      auto tmp9 = at::vec::minimum(tmp8, vec_max_val);
      _store(tmp_out + col, tmp9);
    }
    for (long col = vec_size * (N / vec_size); col < N; col++) {
      auto sum_b = sum_b_ptr[col];
      auto tmp0 = tmp_in[col];
      auto tmp1 = tmp0 - sum_b;
      auto tmp2 = tmp1 - sum_a;
      auto tmp3 = tmp2 + beta1;
      auto tmp4 = (float) tmp3;
      auto tmp5 = tmp4 * alpha;
      auto tmp6 = std::nearbyint(tmp5);
      auto tmp7 = tmp6 + beta2_float;
      auto tmp8 = std::max(tmp7, min_val);
      auto tmp9 = std::min(tmp8, max_val);
      tmp_out[col] = tmp9;
    }
  }
}

template <typename scalar_t>
inline void _int_sum_b_contiguous_kernel_helper(
    const scalar_t* in,
    int32_t* out,
    const int& N,
    const int32_t& scale) {
  const int32_t vec_size = at::vec::Vectorized<int32_t>::size();
  int32_t tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<int32_t>(tmp_sum);
  for (long i = 0; i < vec_size * (N / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(in + i);
    auto tmp1 = at::vec::convert<int32_t>(tmp0);
    vec_tmp_sum = vec_tmp_sum + tmp1;
  }
  tmp_sum += vec_tmp_sum.reduce_add();
  for (long i = vec_size * (N / vec_size); i < N; i++) {
    tmp_sum += static_cast<int32_t>(in[i]);
  }
  out[0] = tmp_sum * scale;
}

template <typename scalar_t>
inline void _int_sum_b_contiguous_kernel(
    const scalar_t* in,
    int32_t* out,
    const int& M,
    const int& N,
    const int& ld,
    const int32_t& scale) {
  for (long r = 0; r < M; r += 1) {
    _int_sum_b_contiguous_kernel_helper(in + r * ld, out + r, N, scale);
  }
}

template <typename scalar_t>
inline void _int_sum_a_contiguous_kernel(
    const scalar_t* in,
    int32_t* out,
    const int& M,
    const int& N,
    const int& ld,
    const int32_t& scale) {
  const int32_t vec_size = at::vec::Vectorized<int32_t>::size();
  auto vec_scale = at::vec::Vectorized<int32_t>(scale);
  // initialization with 0
  int32_t zero = 0;
  auto vec_zero = at::vec::Vectorized<int32_t>(zero);
  for (long i = 0; i < vec_size * (M / vec_size); i += vec_size) {
    _store(out + i, vec_zero);
  }
  for (long i = vec_size * (M / vec_size); i < M; i++) {
    out[i] = zero;
  }
  // sum
  for (long j = 0; j < N; j++) {
    const scalar_t* tmp_in = in + j * ld;
    for (long i = 0; i < vec_size * (M / vec_size); i += vec_size) {
      auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(tmp_in + i);
      auto tmp1 = at::vec::Vectorized<int32_t>::loadu(out + i);
      auto tmp2 = at::vec::convert<int32_t>(tmp0);
      auto tmp3 = tmp1 + tmp2;
      _store(out + i, tmp3);
    }
    for (long i = vec_size * (M / vec_size); i < M; i++) {
      auto tmp0 = tmp_in[i];
      auto tmp1 = out[i];
      auto tmp2 = static_cast<int32_t>(tmp0);
      auto tmp3 = tmp1 + tmp2;
      out[i] = tmp3;
    }
  }
  // scale
  for (long i = 0; i < vec_size * (M / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out + i);
    auto tmp1 = tmp0 * vec_scale;
    _store(out + i, tmp1);
  }
  for (long i = vec_size * (M / vec_size); i < M; i++) {
    auto tmp0 = out[i];
    auto tmp1 = tmp0 * scale;
    out[i] = tmp1;
  }
}

inline void do_convert_u8_s8(
    unsigned char* src,
    signed char* dst,
    int64_t in_rows,
    int64_t in_cols,
    int64_t ldi,
    int64_t ldo) {
  auto vec_size = at::vec::Vectorized<int16_t>::size();
  auto vec_128 = at::vec::Vectorized<int16_t>(128);
  for (int64_t r = 0; r < in_rows; r++) {
    const unsigned char* tmp_src = src + r * ldi;
    signed char* tmp_dst = dst + r * ldo;
    for (int64_t c = 0; c < vec_size * (in_cols / vec_size); c += vec_size) {
      auto tmp0 = at::vec::Vectorized<unsigned char>::loadu(tmp_src + c, vec_size);
      auto tmp1 = at::vec::convert<int16_t>(tmp0);
      auto tmp2 = tmp1 - vec_128;
      auto tmp3 = at::vec::convert<signed char>(tmp2);
      _store(tmp_dst + c, tmp3, vec_size);
    }
    for (int64_t c = vec_size * (in_cols / vec_size); c < in_cols; c++) {
      auto tmp0 = tmp_src[c];
      auto tmp1 = (int16_t) tmp0;
      auto tmp2 = tmp1 - 128;
      auto tmp3 = (signed char) tmp2;
      tmp_dst[c] = tmp3;
    }
  }
}

template <typename scalar_t>
inline void do_transpose(
    scalar_t* src,
    scalar_t* dst,
    int64_t in_rows,
    int64_t in_cols,
    int64_t ldi,
    int64_t ldo) {
  for (int64_t r=0; r<in_rows; r++) {
    for (int64_t c=0; c<in_cols; c++) {
      *(dst + c * ldo + r) = *(src + r * ldi + c);
    }
  }
}

template <typename scalar_t>
inline void do_copy(
    scalar_t* src,
    scalar_t* dst,
    int64_t in_rows,
    int64_t in_cols,
    int64_t ldi,
    int64_t ldo) {
  for (int64_t r=0; r<in_rows; r++) {
    for (int64_t c=0; c<in_cols; c++) {
      *(dst + r * ldo + c) = *(src + r * ldi + c);
    }
  }
}

template <typename scalar_t>
inline void pad_remain_row_col(
    scalar_t* value_ptr,
    int rows,
    int cols,
    int prows,
    int pcols,
    int ldi,
    scalar_t pad_val=0) {
  auto psize = pcols - cols;
  if (psize == 0 && prows == rows) {
    return;
  }
  const int32_t vec_size = at::vec::Vectorized<scalar_t>::size();
  auto pad = at::vec::Vectorized<scalar_t>(pad_val);
  if (psize > 0) {
    for (int i = 0; i < rows; i++) {
      int j = 0;
      for (; j < psize - (psize % vec_size); j += vec_size) {
        pad.store(value_ptr + i * ldi + cols + j);
      }
      if (j < psize) {
        pad.store(value_ptr + i * ldi + cols + j, psize - j);
      }
    }
  }

  for (int i = rows; i < prows; i++) {
    int j = 0;
    for (; j < pcols - (pcols % vec_size); j += vec_size) {
      pad.store(value_ptr + i * ldi + j);
    }
    if (j < pcols) {
      pad.store(value_ptr + i * ldi + j, pcols - j);
    }
  }
}

template <typename scalar_t>
inline void copy_value_with_pad(
    scalar_t* value_ptr,
    scalar_t* dst_ptr,
    int rows,
    int cols,
    int prows,
    int pcols,
    int ldi,
    scalar_t pad_val=0) {
  const int32_t vec_size = at::vec::Vectorized<scalar_t>::size();
  auto pad = at::vec::Vectorized<scalar_t>(pad_val);
  int i = 0;
  for (; i < rows; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(dst_ptr + i * pcols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(dst_ptr + i * pcols + j, cols - j);
    }

    // col padding
    auto psize = pcols - cols;
    if (psize > 0) {
      int pj = 0;
      for (; pj < psize - (psize % vec_size); pj += vec_size) {
        pad.store(dst_ptr + i * pcols + cols + pj);
      }
      if (pj < psize) {
        pad.store(dst_ptr + i * pcols + cols + pj, psize - pj);
      }
    }
  }

  // row padding
  for (; i < prows; i++) {
    int j = 0;
    for (; j < pcols - (pcols % vec_size); j += vec_size) {
      pad.store(dst_ptr + i * pcols + j);
    }
    if (j < pcols) {
      pad.store(dst_ptr + i * pcols + j, pcols - j);
    }

  }

}

// UINT8 - u8u8s32
template <typename scalar_t, typename mask_t, int64_t q_split_size, int64_t kv_split_size>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char>, void>
sdpa_int8_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    at::Tensor& attention_mask,
    double scale,
    int32_t q_zp,
    float q_scale,
    int32_t k_zp,
    float k_scale,
    int32_t v_zp,
    float v_scale,
    int32_t a_zp,
    float a_scale,
    int32_t o_zp,
    float o_scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  const auto accumulate_dtype = at::kFloat;

  using accum_t = float;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale);
  int block_64 = 64;
  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_sdpa: Q/K/V should have the same head size");
  TORCH_CHECK(
      kv_split_size % block_64 == 0, "kv_split_size is not divisble by ", block_64);

  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);


  bool has_attn_mask = attention_mask.defined() && attention_mask.numel();
  if (has_attn_mask) {
    reshape_attn_mask_to_4d(attention_mask, batchSize, num_head, qSize, kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t mStrideB =
      (attention_mask.defined() && attention_mask.size(0) > 1)
      ? attention_mask.stride(0)
      : 0;
  int64_t mStrideH =
      (attention_mask.defined() && attention_mask.size(1) > 1)
      ? attention_mask.stride(1)
      : 0;
  int64_t mStrideM =
      (attention_mask.defined() && attention_mask.size(2) > 1)
      ? attention_mask.stride(2)
      : 0;
  int64_t mStrideN =
      (attention_mask.defined() && attention_mask.size(3) > 1)
      ? attention_mask.stride(3)
      : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t qTail = (qSize - 1) % qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  int64_t rndHeadSize = (headSize + block_64 - 1L) / block_64 * block_64;
  // one of 16, 32, 48, 64
  auto select_tail_tail_block_size = [](int64_t size) -> int64_t {
    if (size == 0) {
      return 0;
    } else if (size <= 16) {
      return 16;
    } else if (size <= 32) {
      return 32;
    } else if (size <= 48) {
      return 48;
    } else {
      return 64;
    }
  };
  int64_t kv_tail_tail_block_size = select_tail_tail_block_size(kvTail % block_64);
  int64_t rndkvSplitSize = (kvSplitSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvTail = (kvTail + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSize = kv_split_size > kvSize ? rndkvTail : rndkvSplitSize * kvSlice + rndkvTail;

  bool av_gemm_K_mul4 = kvSplitSize % 4 == 0;
  int av_gemm_K_padding = av_gemm_K_mul4 ? 0 : 4 - kvSplitSize % 4;
  // // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int av_gemm_K = kvSplitSize + av_gemm_K_padding;
  bool av_gemm_K_tail_mul4 = kvTail % 4 == 0;
  int av_gemm_K_tail_padding = av_gemm_K_tail_mul4 ? 0 : 4 - kvTail % 4;
  int av_gemm_K_tail = kvTail + av_gemm_K_tail_padding;

  auto u8_dt = at::ScalarType::Byte;
  auto s8_dt = at::ScalarType::Int;
  auto f32_dt = at::ScalarType::Float;

  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  mask_t* mask_data = attention_mask.defined()
      ? attention_mask.data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();

  // Create tpp kernels for Query @ Key
  bool headSize_mul4 = headSize % 4 == 0;
  // If K of Gemm is not even, use mkl gemm instead of tpp for BF16
  int qk_gemm_K_padding = headSize_mul4 ? 0 : 4 - headSize % 4;
  int qk_gemm_K = headSize + qk_gemm_K_padding;

  std::vector<std::pair<int64_t, int64_t>> A_B_offsets(1);
  A_B_offsets[0] = std::make_pair(0, 0);

  std::vector<std::pair<int64_t, int64_t>> A_B_offsets_batch(kvSlice);
  for (auto s=0; s<kvSlice; s++) {
    A_B_offsets_batch[s] = std::make_pair(s * qSplitSize * av_gemm_K, s * av_gemm_K * rndHeadSize);
  }

  int64_t total_size_uint8_per_thread =
    /* qk */ kvSlice * qSplitSize * rndkvSplitSize * 4 +
    /* qk_local  */ kvSlice * av_gemm_K * 4 +
    /* qk_reduce  */ kvSlice * qSplitSize * av_gemm_K +
    /* qk_s32   */ qSplitSize * rndkvSplitSize * 4 +
    /* dst_s32  */ qSplitSize * rndHeadSize * 4 +
    /* softmax_sum   */ qSplitSize * 4 +
    /* query_sum     */ qSplitSize * 4 +
    /* attention_sum */ qSplitSize * 4 +
    /* softmax max */ qSplitSize * 4 +
    /* query_padding_data */ qSplitSize * qk_gemm_K;

  at::Tensor total_buf = at::empty(
      {num_thread, total_size_uint8_per_thread},
      query.options()).zero_();
  scalar_t* total_buf_data = total_buf.data_ptr<scalar_t>();

  int64_t kv_sum_size_per_BH =
    /* key_sum */ kvSize +
    /* value_sum */ headSize;

  at::Tensor kv_sum_buf = at::empty(
      {batchSize, num_head, kv_sum_size_per_BH},
      query.options().dtype(at::kInt)).zero_();
  int32_t* kv_sum_buf_data = kv_sum_buf.data_ptr<int32_t>();

  int64_t kv_reorder_size_per_BH =
    /* key_t_reorder */ qk_gemm_K * rndkvSize +
    /* value_t_reorder */ kvSlice * av_gemm_K * rndHeadSize;

  at::Tensor kv_reorder_buf = at::empty(
      {batchSize, num_head, kv_reorder_size_per_BH},
      query.options()).zero_();
  scalar_t* kv_reorder_buf_data = kv_reorder_buf.data_ptr<scalar_t>();
  scalar_t* key_reorder_ptr = kv_reorder_buf_data;
  scalar_t* value_reorder_ptr = kv_reorder_buf_data + batchSize * num_head * qk_gemm_K * rndkvSize;

  // sum k and v
  at::parallel_for(
      0, batchSize * num_head, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head);
        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          int32_t* kv_sum_ptr = kv_sum_buf_data
              + i * num_head * kv_sum_size_per_BH
              + j * kv_sum_size_per_BH;
          int32_t* k_sum_ptr = kv_sum_ptr;
          int32_t* v_sum_ptr = kv_sum_ptr + kvSize;
          if (q_zp == 0) {
            fill_stub(k_sum_ptr, static_cast<int32_t>(0), kvSize);
          } else {
            _int_sum_b_contiguous_kernel(k_data + i * kStrideB + j * kStrideH,
              k_sum_ptr,
              kvSize, headSize, kStrideN, q_zp);
          }
          if (a_zp == 0) {
            fill_stub(v_sum_ptr, static_cast<int32_t>(0), headSize);
          } else {
            _int_sum_a_contiguous_kernel(v_data + i * vStrideB + j * vStrideH,
              v_sum_ptr,
              headSize, kvSize, vStrideN, a_zp);
          }
        // Move to the next query
        at::native::data_index_step(i, batchSize, j, num_head);
      }
    });

  // packing
  at::parallel_for(
    0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
      int64_t i = 0, j = 0, l = 0, n = 0;
      at::native::data_index_init(
          begin, i, batchSize, j, num_head, l, kvSlice);
      uint8_t* B_blocked_xform_u8 = new uint8_t[std::max(qk_gemm_K, av_gemm_K) * block_64];
      for (const auto z : c10::irange(begin, end)) {
        (void)z; // Suppress unused variable
        n = l * kvSplitSize;
        if (n + kvSplitSize < kvSize) {
          for (int64_t b = 0; b < rndkvSplitSize; b += block_64) {
            bool tail = kvSplitSize - b < block_64;
            do_transpose(
                k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                B_blocked_xform_u8,
                tail ? kvSplitSize - b : block_64,
                headSize,
                kStrideN,
                block_64);
            if (!headSize_mul4) {
              pad_remain_row_col(
                  B_blocked_xform_u8,
                  headSize,
                  block_64,
                  qk_gemm_K,
                  block_64,
                  block_64
                );
            }
            at::native::cpublas::pack(
                    qk_gemm_K, // K
                    block_64, // N
                    block_64, // ld_in
                    block_64, // ld_out
                    u8_dt, // dt_in
                    u8_dt, // dt_out
                    B_blocked_xform_u8,
                    key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                      b * qk_gemm_K);
          }
          // split headSize to block_64, block_64, block_64 ...
          // [kvSplitSize, headSize] -> [av_gemm_K,  block_64 ...]
          for (int64_t b = 0; b < rndHeadSize; b += block_64) {
            at::native::cpublas::pack(
                    av_gemm_K,
                    block_64,
                    vStrideN, // block_64,
                    block_64,
                    u8_dt,
                    u8_dt,
                    v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                    value_reorder_ptr +
                      i * num_head * kvSlice * av_gemm_K * rndHeadSize +
                      j * kvSlice * av_gemm_K * rndHeadSize + n * rndHeadSize +
                      av_gemm_K * b);
          }
        } else {
          // tail
          auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
          int64_t b = 0;
          while (b < rndkvTail) {
            bool tail = kvTail - b < block_size;
            do_transpose(
                k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                B_blocked_xform_u8,
                tail ? kvTail - b : block_size,
                headSize,
                kStrideN,
                block_size);
            if (!headSize_mul4) {
              pad_remain_row_col(
                  B_blocked_xform_u8,
                  headSize,
                  block_size,
                  qk_gemm_K,
                  block_size,
                  block_size
                );
            }
            // Pack
            if (block_size == block_64) {
              at::native::cpublas::pack(
                      qk_gemm_K,
                      block_64,
                      block_64,
                      block_64,
                      u8_dt,
                      u8_dt,
                      B_blocked_xform_u8,
                      key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                        j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                        b * qk_gemm_K);
            } else {
              at::native::cpublas::pack(
                      qk_gemm_K,
                      kv_tail_tail_block_size,
                      kv_tail_tail_block_size,
                      kv_tail_tail_block_size,
                      u8_dt,
                      u8_dt,
                      B_blocked_xform_u8,
                      key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                      b * qk_gemm_K);
            }
            b += block_size;
            block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
          }
          // split headSize to block_64, block_64, block_64 ...
          // [kvTail, headSize] -> [av_gemm_K_tail,  block_64 ...]
          for (int64_t b = 0; b < headSize; b += block_64) {
            at::native::cpublas::pack(
                    av_gemm_K,
                    block_64,
                    vStrideN, // block_64,
                    block_64,
                    u8_dt,
                    u8_dt,
                    v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                    value_reorder_ptr +
                      i * num_head * kvSlice * av_gemm_K * rndHeadSize +
                      j * kvSlice * av_gemm_K * rndHeadSize + n * rndHeadSize +
                      av_gemm_K * b);
          }
        }
        // Move to the next query
        at::native::data_index_step(i, batchSize, j, num_head, l, kvSlice);
      }
    });

  at::parallel_for(
      0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        scalar_t* total_buf_ptr = total_buf_data + ompIdx * total_size_uint8_per_thread;
        int32_t offset = 0;
        accum_t* qk_data = reinterpret_cast<accum_t*>(total_buf_ptr);
        offset += kvSlice * qSplitSize * rndkvSplitSize * 4;
        accum_t* qk_local_data = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += kvSlice * av_gemm_K * 4;
        scalar_t* qk_reduced_data = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += kvSlice * qSplitSize * av_gemm_K;
        int32_t* qk_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndkvSplitSize * 4;
        int32_t* dst_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndHeadSize * 4;
        accum_t* sfm_sum_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* q_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* a_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        accum_t* sfm_max_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        scalar_t* query_t_padding_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable

          int32_t* kv_sum_ptr = kv_sum_buf_data
              + i * num_head * kv_sum_size_per_BH
              + j * kv_sum_size_per_BH;
          int32_t* k_sum_ptr = kv_sum_ptr;
          int32_t* v_sum_ptr = kv_sum_ptr + kvSize;

          // sdpa core
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize sum and max
          fill_stub(
              sfm_sum_ptr, static_cast<accum_t>(0), qSplitSize);
          fill_stub(
              a_sum_ptr, static_cast<int32_t>(0), qSplitSize);
          fill_stub(
              sfm_max_ptr, static_cast<accum_t>(-std::numeric_limits<accum_t>::infinity()), qSplitSize);
          int64_t num_keys =
              is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
          copy_value_with_pad(
              q_data + i * qStrideB + j * qStrideH + m * qStrideM,
              query_t_padding_ptr,
              qBlockSize,
              headSize,
              qBlockSize,
              qk_gemm_K,
              qStrideM);

          if (k_zp != 0) {
            _int_sum_b_contiguous_kernel(q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                  q_sum_ptr, qBlockSize, headSize, qStrideM, k_zp);
          } else {
            fill_stub(
              q_sum_ptr, static_cast<int32_t>(0), qSplitSize);
          }
          const int64_t rkvSlice = (num_keys - 1) / kvSplitSize + 1;
          for (int64_t l = 0; l < rkvSlice; l++) {
            int64_t n = l * kvSplitSize;
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            // Calculate sums for dequant compensation item
            if (qBlockSize == qSplitSize) {
              // q main
              if (n + kvSplitSize < kvSize) {
                // k main
                for (int64_t b = 0; b < kvSplitSize; b += block_64) {
                  at::native::cpublas::brgemm(
                        qSplitSize, block_64, qk_gemm_K,
                        1, //batch_size
                        qk_gemm_K, // lda
                        block_64, //ldb
                        rndkvSplitSize, //ldc,
                        false,
                        query_t_padding_ptr,
                        key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                          j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                          b * qk_gemm_K,
                        qk_s32_data + b,
                        A_B_offsets);
                }
              } else {
                auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
                int64_t b = 0;
                while (b < kvTail) {
                  if (block_size == block_64) {
                    at::native::cpublas::brgemm(
                          qSplitSize, block_64, qk_gemm_K,
                          1, //batch_size
                          qk_gemm_K, // lda
                          block_64, //ldb
                          rndkvTail, //ldc
                          false,
                          query_t_padding_ptr,
                          key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                            j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                            b * qk_gemm_K,
                          qk_s32_data + b,
                          A_B_offsets);
                  } else {
                    at::native::cpublas::brgemm(
                          qSplitSize, kv_tail_tail_block_size, qk_gemm_K,
                          1, //batch_size
                          qk_gemm_K, // lda
                          kv_tail_tail_block_size, //ldb
                          rndkvTail, //ldc
                          false,
                          query_t_padding_ptr,
                          key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                            j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                            b * qk_gemm_K,
                          qk_s32_data + b,
                          A_B_offsets);
                  }
                  b += block_size;
                  block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
                }
              }
            } else {
              if (n + kvSplitSize < kvSize) {
                for (int64_t b = 0; b < kvSplitSize; b += block_64) {
                  at::native::cpublas::brgemm(
                        qTail, block_64, qk_gemm_K,
                        1, //batch_size
                        qk_gemm_K,// lda
                        block_64, //ldb
                        rndkvSplitSize, //ldc
                        false,
                        query_t_padding_ptr,
                        key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                          j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                          b * qk_gemm_K,
                        qk_s32_data + b,
                        A_B_offsets);
                }
              } else {
                // k tail
                auto block_size = kvTail >= block_64 ? block_64 : kv_tail_tail_block_size;
                int64_t b = 0;
                while (b < kvTail) {
                  if (block_size == block_64) {
                    at::native::cpublas::brgemm(
                          qTail, block_64, qk_gemm_K,
                          1, //batch_size
                          qk_gemm_K, // lda
                          block_64, //ldb
                          rndkvTail, //ldc
                          false,
                          query_t_padding_ptr,
                          key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                            j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                            b * qk_gemm_K,
                          qk_s32_data + b,
                          A_B_offsets);
                  } else {
                    at::native::cpublas::brgemm(
                          qSplitSize, kv_tail_tail_block_size, qk_gemm_K,
                          1, //batch_size
                          qk_gemm_K, // lda
                          kv_tail_tail_block_size, //ldb
                          rndkvTail, //ldc
                          false,
                          query_t_padding_ptr,
                          key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                            j * qk_gemm_K * rndkvSize + n * qk_gemm_K +
                            b * qk_gemm_K,
                          qk_s32_data + b,
                          A_B_offsets);
                  }
                b += block_size;
                block_size = (kvTail - b) >= block_64 ? block_64 : kv_tail_tail_block_size;
                }
              }
            }

            // do dequant compensation, add mask, max reduce for softmax, and convert qk from s32 to fp32
            int64_t rndkvBlockSize = kvBlockSize == kvSplitSize ? rndkvSplitSize : rndkvTail;
            accum_t* qk_block_data = qk_data + l * qSplitSize * rndkvSplitSize;
            if (has_attn_mask) {
              mask_t* mask_data_offset = mask_data + i * mStrideB + j * mStrideH + m * mStrideM + (mStrideN == 0 ? 0 : n);
              _dequant_mask_max_fusion_kernel(
                qk_s32_data, //in
                mask_data_offset, //mask_ptr
                q_sum_ptr, //sum_a_ptr
                k_sum_ptr + n, //sum_b_ptr
                qBlockSize, //M
                kvBlockSize, //N
                rndkvBlockSize, //ldi
                mStrideM, //ldm
                rndkvSplitSize,//kvBlockSize, //ldo
                q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                qk_block_data, //out
                sfm_max_ptr // sfm_max_ptr
              );
            } else {
              _dequant_max_fusion_kernel(
                qk_s32_data, //in
                q_sum_ptr, //sum_a_ptr
                k_sum_ptr + n, //sum_b_ptr
                qBlockSize, //M
                kvBlockSize, //N
                rndkvBlockSize, //ldi
                rndkvSplitSize,//kvBlockSize, //ldo
                q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                qk_block_data, //out
                sfm_max_ptr // sfm_max_ptr
              );  
            }
          }
          // sub max, exp, sum reduce, div sum for softmax
          // and quant
          // and sum for attention
          if (v_zp == 0) {
            _sub_exp_sum_div_quant_fusion_kernel(
              qk_data, //in
              qBlockSize, //M
              kvSplitSize, //N_step
              rkvSlice, //NSlices
              qSplitSize * rndkvSplitSize, //ldi
              qSplitSize * av_gemm_K, //ldo
              kvSize, //kvSize
              rndkvSplitSize, //rndkvSplitSize
              av_gemm_K, //av_gemm_K
              a_zp, // zp_a=beta1
              a_scale, // scale_a=alpha
              qk_local_data, //local
              qk_reduced_data, //out
              sfm_max_ptr, //sfm_max_ptr
              sfm_sum_ptr //sfm_sum_ptr
            );
          } else {
            _sub_exp_sum_div_quant_sum_fusion_kernel(
              qk_data, //in
              qBlockSize, //M
              kvSplitSize, //N_step
              rkvSlice, //NSlice
              qSplitSize * rndkvSplitSize, //ldi
              qSplitSize * av_gemm_K, //ldo
              kvSize, //kvSize
              rndkvSplitSize, //rndkvSplitSize
              av_gemm_K, //av_gemm_K
              a_zp, // zp_a=beta1
              v_zp, // zp_b=beta2
              a_scale, // scale_a=alpha
              qk_local_data, //local
              qk_reduced_data, //out
              sfm_max_ptr, //sfm_max_ptr
              sfm_sum_ptr, //sfm_sum_ptr
              a_sum_ptr //a_sum_ptr
            );
          }
          // Calculate Softmax(q @ k.T) @ v
          for (int64_t b = 0; b < headSize; b += block_64) {
            at::native::cpublas::brgemm(
                  qSplitSize, block_64, av_gemm_K,
                  kvSlice, //batch_size
                  av_gemm_K, // lda
                  rndHeadSize, //block_64, //ldb
                  rndHeadSize, //ldc
                  false,
                  qk_reduced_data,
                  value_reorder_ptr +
                    i * num_head * kvSlice * av_gemm_K * rndHeadSize +
                    j * kvSlice * av_gemm_K * rndHeadSize + b * av_gemm_K,
                  dst_s32_data + b,
                  A_B_offsets_batch);
          }

          // After the last gemm,
          // do dequant compensation, quant and convert from s32 to int8
          _dequant_quant_fusion_kernel(
            dst_s32_data, //in
            a_sum_ptr, //sum_a_ptr
            v_sum_ptr, //sum_b_ptr
            qBlockSize, //M
            headSize, //N
            rndHeadSize, //ldi
            oStrideM, //ldo
            a_zp * v_zp * kvSize, //zp_a*zp_b*k=beta1
            o_zp, //zp_c=beta2
            a_scale * v_scale / o_scale, //scale_a*scale_b/scale_c=alpha
            out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
          );
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
  // Once all computations are done, need to release HW context.
  at::native::cpublas::brgemm_release();
}

#define AT_DISPATCH_MASK_TYPES(TYPE, NAME, ...)            \
  AT_DISPATCH_SWITCH(                                      \
      TYPE,                                                \
      NAME,                                                \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Bool, mask_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Float, mask_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Double, mask_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::BFloat16, mask_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Half, mask_t, __VA_ARGS__))

void sdpa_int8_kernel(
    const at::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    at::Tensor& attn_mask,
    double scale,
    long q_zp,
    double q_scale,
    long k_zp,
    double k_scale,
    long v_zp,
    double v_scale,
    long a_zp,
    double a_scale,
    long o_zp,
    double o_scale) {
  int64_t batchSize = query.size(0);
  int64_t num_head = query.size(1);
  int64_t q_seq_len = query.size(2);

  TORCH_CHECK(query.scalar_type() == c10::kByte);
  if (!attn_mask.defined()) {
    if (q_seq_len >= 768) {
      sdpa_int8_kernel_impl<unsigned char, float, 256, 64>(
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_zp, q_scale,
        k_zp, k_scale,
        v_zp, v_scale,
        a_zp, a_scale,
        o_zp, o_scale);
    } else if (q_seq_len >= 192) {
      sdpa_int8_kernel_impl<unsigned char, float, 64, 64>(
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_zp, q_scale,
        k_zp, k_scale,
        v_zp, v_scale,
        a_zp, a_scale,
        o_zp, o_scale);
    } else {
      sdpa_int8_kernel_impl<unsigned char, float, 32, 64>(
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_zp, q_scale,
        k_zp, k_scale,
        v_zp, v_scale,
        a_zp, a_scale,
        o_zp, o_scale);
    }
  } else {
    AT_DISPATCH_MASK_TYPES(attn_mask.scalar_type(), "sdpa_mask", [&]() {
      if (q_seq_len >= 768) {
        sdpa_int8_kernel_impl<unsigned char, mask_t, 256, 64>(
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      } else if (q_seq_len >= 192) {
        sdpa_int8_kernel_impl<unsigned char, mask_t, 64, 64>(
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      } else {
        sdpa_int8_kernel_impl<unsigned char, mask_t, 32, 64>(
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_zp, q_scale,
          k_zp, k_scale,
          v_zp, v_scale,
          a_zp, a_scale,
          o_zp, o_scale);
      }
    });
  }
}

at::Tensor _scaled_dot_product_int8_cpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& attn_mask,
    // const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    double scale,
    // std::optional<double> scale,
    int64_t q_zp,
    double q_scale,
    int64_t k_zp,
    double k_scale,
    int64_t v_zp,
    double v_scale,
    int64_t a_zp,
    double a_scale,
    int64_t o_zp,
    double o_scale) {
  const auto dtype = query.scalar_type();
  TORCH_CHECK(!query.is_nested() && !key.is_nested() && !value.is_nested(),
    "_scaled_dot_product_int8_cpu: Only accept plain inputs");
  TORCH_CHECK(!is_causal,
    "_scaled_dot_product_int8_cpu: is_causal not supported.");
  TORCH_CHECK(dtype == at::ScalarType::Byte,
    "_scaled_dot_product_int8_cpu: Expected data type be U8, but got ", dtype, " instead.");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
    "_scaled_dot_product_int8_cpu: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_CHECK(dropout_p == 0.0,
    "_scaled_dot_product_int8_cpu: Currently do not support dropout > 0");
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
    "_scaled_dot_product_int8_cpu: Q/K/V should have the same head size");
  TORCH_CHECK(!attn_mask.defined() ||
          attn_mask.scalar_type() == at::kFloat ||
          attn_mask.scalar_type() == at::kBFloat16,
    "_scaled_dot_product_int8_cpu: Expected attention mask be float or bf16");
  TORCH_CHECK(!attn_mask.defined() ||
          (attn_mask.dim() == 2 || attn_mask.dim() == 4),
    "_scaled_dot_product_int8_cpu: Attention mask dim in {2, 4}");

  at::Tensor output = at::empty_like(query, query.options()).transpose(1, 2);
  sdpa_int8_kernel(output, query, key, value,
      dropout_p, is_causal, attn_mask, scale,
      q_zp, q_scale,
      k_zp, k_scale,
      v_zp, v_scale,
      a_zp, a_scale,
      o_zp, o_scale);

  return output.transpose(1, 2);
}


} // anonymous namespace

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::scaled_dot_product_int8", &_scaled_dot_product_int8_cpu);
}

// } // at::native
} // namespace torchao