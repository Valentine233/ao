from typing import Any, List, Optional

import sympy
import torch
from torch._inductor.ir import FixedLayout, get_fill_order, TensorBox, ExternKernelAlloc
from torch._inductor.lowering import expand, register_lowering, make_fallback
from torch._inductor.kernel.flex_attention import construct_strides, maybe_realize
# from torchao.ops import scaled_dot_product_int8
from torch import _scaled_dot_product_int8

class Int8SDPA(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=_scaled_dot_product_int8,
            cpp_kernel_name="aoti_torch_cpu__scaled_dot_product_int8",
            # op_overload=torch.ops.torchao.scaled_dot_product_int8.default,
        )

    # def codegen(self, wrapper):
    #     wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
    #     super().codegen(wrapper)

    @classmethod
    def create(
        cls,
        query: TensorBox,
        key: TensorBox,
        value: TensorBox,
        attn_mask: Optional[TensorBox],
        dropout: float,
        is_causal: bool,
        scale: float,
        q_zp: Optional[int] = 0,
        q_scale: Optional[float] = 1.0,
        k_zp: Optional[int] = 0,
        k_scale: Optional[float] = 1.0,
        v_zp: Optional[int] = 0,
        v_scale: Optional[float] = 1.0,
        a_zp: Optional[int] = 0,
        a_scale: Optional[float] = 1.0,
        o_zp: Optional[int] = 0,
        o_scale: Optional[float] = 1.0,
    ):
        (
            query,
            key,
            value,
            attn_mask,
        ) = maybe_realize(
            [
                query,
                key,
                value,
                attn_mask,
            ]
        )

        if (
            query.get_dtype() is not torch.uint8
            or key.get_dtype() is not torch.uint8
            or value.get_dtype() is not torch.uint8
        ):
            raise NotImplementedError(
                "Only `torch.uint8` is supported in Int8 SDPA template for CPU device. "
                f"Found input tensors are `{query.get_dtype()}`,`{key.get_dtype()}`,`{value.get_dtype()}`."
            )

        # Construct output layout with strides matching the query.
        out_size = query.get_size()
        fill_order = get_fill_order(query.get_stride())
        out_strides = construct_strides(out_size, fill_order)

        layout = FixedLayout(
            query.get_device(),
            query.get_dtype(),
            out_size,
            stride=[sympy.sympify(s) for s in out_strides],
        )

        constant_args = [
            dropout,
            is_causal,
            scale,
            q_zp,
            q_scale,
            k_zp,
            k_scale,
            v_zp,
            v_scale,
            a_zp,
            a_scale,
            o_zp,
            o_scale,
        ]

        inputs = [
            query,
            key,
            value,
        ]
        if attn_mask is not None:
            inputs.append(attn_mask)
        else:
            constant_args.insert(0, attn_mask)

        return Int8SDPA(
            layout=layout,
            inputs=inputs,
            constant_args=constant_args,
        )

def register_int8_sdpa():
    # @register_lowering(scaled_dot_product_int8, type_promotion_kind=None)
    @register_lowering(_scaled_dot_product_int8, type_promotion_kind=None)
    def int8_sdpa(
        query: TensorBox,
        key: TensorBox,
        value: TensorBox,
        attn_mask: Optional[TensorBox],
        dropout: float,
        is_causal: bool,
        scale: float,
        q_zp: Optional[int] = 0,
        q_scale: Optional[float] = 1.0,
        k_zp: Optional[int] = 0,
        k_scale: Optional[float] = 1.0,
        v_zp: Optional[int] = 0,
        v_scale: Optional[float] = 1.0,
        a_zp: Optional[int] = 0,
        a_scale: Optional[float] = 1.0,
        o_zp: Optional[int] = 0,
        o_scale: Optional[float] = 1.0,
    ):
        return TensorBox.create(
            Int8SDPA.create(
                query,
                key,
                value,
                attn_mask,
                dropout,
                is_causal,
                scale,
                q_zp,
                q_scale,
                k_zp,
                k_scale,
                v_zp,
                v_scale,
                a_zp,
                a_scale,
                o_zp,
                o_scale,
            )
        )

register_int8_sdpa()

# def int8_sdpa_lowering(
#     query: TensorBox,
#     key: TensorBox,
#     value: TensorBox,
#     inv_scale: float,
#     attn_mask: Optional[TensorBox],
#     q_zp: Optional[int] = 0,
#     q_scale: Optional[float] = 1.0,
#     k_zp: Optional[int] = 0,
#     k_scale: Optional[float] = 1.0,
#     v_zp: Optional[int] = 0,
#     v_scale: Optional[float] = 1.0,
#     a_zp: Optional[int] = 0,
#     a_scale: Optional[float] = 1.0,
#     o_zp: Optional[int] = 0,
#     o_scale: Optional[float] = 1.0,
# ) -> TensorBox:
#     (
#         query,
#         key,
#         value,
#         attn_mask,
#     ) = maybe_realize(
#         [
#             query,
#             key,
#             value,
#             attn_mask,
#         ]
#     )

#     if (
#         query.get_dtype() is not torch.uint8
#         or key.get_dtype() is not torch.uint8
#         or value.get_dtype() is not torch.uint8
#     ):
#         raise NotImplementedError(
#             "Only `torch.uint8` is supported in Int8 SDPA template for CPU device. "
#             f"Found input tensors are `{query.get_dtype()}`,`{key.get_dtype()}`,`{value.get_dtype()}`."
#         )

#     # Construct output layout with strides matching the query.
#     out_size = query.get_size()
#     fill_order = get_fill_order(query.get_stride())
#     out_strides = construct_strides(out_size, fill_order)

#     layout = FixedLayout(
#         query.get_device(),
#         query.get_dtype(),
#         out_size,
#         stride=[sympy.sympify(s) for s in out_strides],
#     )
#     _choices: List[Any] = []
#     input_nodes = [query, key, value]
#     if attn_mask is not None:
#         input_nodes.append(attn_mask)

#     CppInt8SdpaTemplate.add_choices(
#         choices=_choices,
#         input_nodes=input_nodes,
#         layout=layout,
#         scale=1.0 / inv_scale,
#         q_zp=q_zp,
#         q_scale=q_scale,
#         k_zp=k_zp,
#         k_scale=k_scale,
#         v_zp=v_zp,
#         v_scale=v_scale,
#         a_zp=a_zp,
#         a_scale=a_scale,
#         o_zp=o_zp,
#         o_scale=o_scale,
#     )
#     inputs_for_autotuning = [
#         query,
#         key,
#         value,
#     ]
#     return autotune_select_algorithm(
#         "int8_sdpa",
#         _choices,
#         inputs_for_autotuning,
#         layout,
#     )


# int8_sdpa_lowering._inductor_lowering_function = True  # type: ignore[attr-defined]
