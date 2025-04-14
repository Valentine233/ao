from typing import Any, List, Optional

import sympy
import torch
from torch._inductor.ir import FixedLayout, get_fill_order, TensorBox, ExternKernelAlloc
from torch._inductor.lowering import expand, register_lowering, make_fallback
from torch._inductor.kernel.flex_attention import construct_strides, maybe_realize
from torch._inductor.utils import use_max_autotune
from torch._inductor.select_algorithm import autotune_select_algorithm, ExternKernelChoice
from .codegen.cpp_int8_sdpa_template import CppInt8SdpaTemplate

op_int8_sdpa = ExternKernelChoice(torch.ops.torchao.scaled_dot_product_int8.default,
                                  "torchao::scaled_dot_product_int8",
                                  has_out_variant=False,
                                  use_fallback_kernel=True,
                                  op_overload=torch.ops.torchao.scaled_dot_product_int8.default)

def register_int8_sdpa():
    @register_lowering(torch.ops.torchao.scaled_dot_product_int8.default, type_promotion_kind=None)
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
    ) -> TensorBox:

        choices: list[ChoiceCaller] = []

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
        input_nodes = [query, key, value]
        if attn_mask is not None:
            input_nodes.append(attn_mask)

        if torch._C._cpu._is_avx512_supported():
            CppInt8SdpaTemplate.add_choices(
                choices=choices,
                input_nodes=input_nodes,
                layout=layout,
                scale=scale,
                q_zp=q_zp,
                q_scale=q_scale,
                k_zp=k_zp,
                k_scale=k_scale,
                v_zp=v_zp,
                v_scale=v_scale,
                a_zp=a_zp,
                a_scale=a_scale,
                o_zp=o_zp,
                o_scale=o_scale,
            )

        if len(choices) == 0:
            choices.append(
                op_int8_sdpa.bind(
                    input_nodes=input_nodes,
                    layout=layout,
                    scale=scale,
                    q_zp=q_zp,
                    q_scale=q_scale,
                    k_zp=k_zp,
                    k_scale=k_scale,
                    v_zp=v_zp,
                    v_scale=v_scale,
                    a_zp=a_zp,
                    a_scale=a_scale,
                    o_zp=o_zp,
                    o_scale=o_scale,
                )
            )

        inputs_for_autotuning = [
            query,
            key,
            value,
        ]

        return autotune_select_algorithm(
            "int8_sdpa",
            choices,
            inputs_for_autotuning,
            layout,
        )


register_int8_sdpa()
