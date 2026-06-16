"""FX op handlers for the Alloy torch.compile backend."""

from __future__ import annotations

import operator
import torch
import alloy_torch.custom_ops  # noqa: F401
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.buffer_ops import k_logical_not
from alloy_torch.ops.attention import (
    _attention_cache_handler,
    _attention_kv_update_handler,
    _attention_kv_update_multi_bidir_handler,
    _attention_kv_update_multi_handler,
    _spec_kv_write_handler,
    _attention_prefill_cold_handler,
    _attention_prefill_warm_handler,
    _scaled_dot_product_attention,
    _scaled_dot_product_flash_attention_for_cpu,
    _scaled_dot_product_flash_attention_for_cpu_backward,
)
from alloy_torch.ops.casting import _copy, _to_copy
from alloy_torch.ops.concat import _cat
from alloy_torch.ops.conv import _convolution
from alloy_torch.ops.delta_net import _linear_attention_update_handler
from alloy_torch.ops.short_conv import _short_conv_gated_handler, _short_conv_update_handler
from alloy_torch.ops.creation import (
    _arange_default,
    _arange_start,
    _arange_start_step,
    _full,
    _full_like,
    _new_empty_strided,
    _ones_like,
    _scalar_tensor,
    _zeros,
)
from alloy_torch.ops.elementwise import (
    _bitwise_and_tensor,
    _bitwise_or_tensor,
    _buf_add,
    _buf_add_scalar,
    _buf_sub,
    _clamp,
    _eq,
    _ne,
    _pow_scalar_tensor,
    _pow_tensor_scalar,
    _reciprocal,
    _where_self,
)
from alloy_torch.ops.indexing import (
    _cache_write_dim2_4d,
    _constant_pad_nd,
    _embedding,
    _gather,
    _index_put,
    _index_put_inplace,
    _index_tensor,
    _scatter_add,
    _scatter_value,
    _select_backward,
    _slice_scatter,
)
from alloy_torch.ops.linalg import (
    _addmm,
    _alloy_batched_dequant_mm_handler,
    _alloy_batched_mm_handler,
    _alloy_dequant_mm_handler,
    _alloy_dequant_silu_handler,
    _alloy_dot_silu_handler,
    _alloy_batched_gguf_q4_k_mm_handler,
    _alloy_batched_gguf_q5_0_mm_handler,
    _alloy_batched_gguf_q6_k_mm_handler,
    _alloy_batched_gguf_q8_0_mm_handler,
    _alloy_gguf_q4_k_embedding_handler,
    _alloy_gguf_q4_k_mm_handler,
    _alloy_gguf_q4_k_silu_handler,
    _alloy_gguf_q4_k_gelu_handler,
    _alloy_mlx_q4_mm_handler,
    _alloy_batched_mlx_q4_mm_handler,
    _alloy_mlx_q4_silu_handler,
    _alloy_mlx_q4_embedding_handler,
    _alloy_gguf_q5_0_embedding_handler,
    _alloy_gguf_q5_0_mm_handler,
    _alloy_gguf_q6_k_embedding_handler,
    _alloy_gguf_q6_k_mm_handler,
    _alloy_gguf_q8_0_embedding_handler,
    _alloy_gguf_q8_0_mm_handler,
    _alloy_gguf_q8_0_silu_handler,
    _bmm,
    _mm,
)
from alloy_torch.ops.kv_quant import _attention_cache_q8_handler
from alloy_torch.ops.moe import _gguf_moe_routed_handler
from alloy_torch.ops.norms import (
    _fused_rms_norm,
    _fused_rms_norm_backward,
    _alloy_gemm_residual_layernorm_handler,
    _alloy_gemm_residual_rmsnorm_handler,
    _native_group_norm,
    _native_layer_norm,
)
from alloy_torch.ops.reductions import (
    _alloy_cross_entropy_bwd_fused_handler,
    _alloy_cross_entropy_fwd_fused_handler,
    _amax_dim,
    _argmax,
    _any_dim,
    _cumsum,
    _log_softmax,
    _mean_dim,
    _softmax,
    _sum_dim,
    _var_correction,
)
from alloy_torch.ops.rope import (
    _fused_rms_norm_rope,
    _fused_rope_apply,
    _fused_rope_apply_backward,
    _rope_table,
)
from alloy_torch.ops.sampling import _sample_categorical_handler
from alloy_torch.ops.views import (
    _alias,
    _expand,
    _repeat,
    _select_int,
    _slice_tensor,
    _split_tensor,
    _split_with_sizes,
    _squeeze_dims,
    _transpose_dims,
    _unfold,
    _unsqueeze,
)


ATEN_TO_ALLOY = {
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default: _scaled_dot_product_flash_attention_for_cpu,
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward.default: _scaled_dot_product_flash_attention_for_cpu_backward,
    torch.ops.aten._softmax.default: _softmax,
    torch.ops.aten._to_copy.default: _to_copy,
    torch.ops.prims.convert_element_type.default: lambda x, dtype: _to_copy(x, dtype=dtype),
    torch.ops.aten._unsafe_view.default: lambda x, shape: x.reshape(shape),
    torch.ops.aten.addmm.default: _addmm,
    torch.ops.aten.alias.default: _alias,
    torch.ops.aten.amax.default: _amax_dim,
    torch.ops.aten.any.dim: _any_dim,
    torch.ops.aten.argmax.default: _argmax,
    torch.ops.aten.arange.default: _arange_default,
    torch.ops.aten.arange.start: _arange_start,
    torch.ops.aten.arange.start_step: _arange_start_step,
    torch.ops.aten.bmm.default: _bmm,
    torch.ops.aten.bitwise_and.Tensor: _bitwise_and_tensor,
    torch.ops.aten.bitwise_or.Tensor: _bitwise_or_tensor,
    torch.ops.aten.cat.default: _cat,
    torch.ops.aten.clone.default: lambda x, *args, **kwargs: x,
    torch.ops.aten.constant_pad_nd.default: _constant_pad_nd,
    torch.ops.aten.convolution.default: _convolution,
    torch.ops.aten.cumsum.default: _cumsum,
    torch.ops.aten.embedding.default: _embedding,
    torch.ops.aten.expand.default: _expand,
    torch.ops.aten.full.default: _full,
    torch.ops.aten.gelu.default: AlloyBuffer.gelu,
    torch.ops.aten._log_softmax.default: _log_softmax,
    torch.ops.aten.logical_not.default: lambda x: k_logical_not(x.contiguous(), N=x.size).reshape(
        x.shape
    ),
    # Qwen3.5 GatedDeltaNet attention-mask path uses logical_and(a, b)
    # on bool masks. `_bitwise_and_tensor` already routes bool×bool
    # through `mask_a * mask_b` which gives the logical-and semantic.
    torch.ops.aten.logical_and.default: _bitwise_and_tensor,
    # gemma4 audio combines padding + sliding-window masks with logical_or.
    torch.ops.aten.logical_or.default: _bitwise_or_tensor,
    # gemma4 audio block-local attention windows K/V with unfold (strided view).
    torch.ops.aten.unfold.default: _unfold,
    # `log1p(x) = log(x + 1)` — used by softplus(dt + dt_bias) inside
    # the GatedDeltaNet selective-scan. AlloyBuffer.__add__ handles the
    # scalar add and .log() the natural log.
    torch.ops.aten.log.default: lambda x: x.log(),
    torch.ops.aten.log1p.default: lambda x: (x + 1.0).log(),
    torch.ops.aten.index_copy.default: lambda x, dim, index, source: _cache_write_dim2_4d(
        x, dim, index, source
    ),
    torch.ops.aten.index_copy_.default: lambda x, dim, index, source: _cache_write_dim2_4d(
        x, dim, index, source, inplace=True
    ),
    torch.ops.aten.index_put.default: _index_put,
    torch.ops.aten.index_put_.default: _index_put_inplace,
    torch.ops.aten.index.Tensor: _index_tensor,
    # --- Unary elementwise ---
    torch.ops.aten.abs.default: AlloyBuffer.abs,
    torch.ops.aten.cos.default: AlloyBuffer.cos,
    torch.ops.aten.erf.default: AlloyBuffer.erf,
    torch.ops.aten.exp.default: AlloyBuffer.exp,
    torch.ops.aten.neg.default: AlloyBuffer.__neg__,
    torch.ops.aten.floor.default: AlloyBuffer.floor,
    torch.ops.aten.relu.default: AlloyBuffer.relu,
    torch.ops.aten.rsqrt.default: AlloyBuffer.rsqrt,
    torch.ops.aten.sigmoid.default: AlloyBuffer.sigmoid,
    torch.ops.aten.sin.default: AlloyBuffer.sin,
    torch.ops.aten.sqrt.default: AlloyBuffer.sqrt,
    # --- Binary elementwise (AlloyBuffer operators) ---
    torch.ops.aten.add.Tensor: _buf_add,
    torch.ops.aten.add.Scalar: _buf_add_scalar,
    torch.ops.aten.mul.Tensor: AlloyBuffer.__mul__,
    torch.ops.aten.mul.Scalar: AlloyBuffer.__mul__,
    torch.ops.aten.sub.Tensor: _buf_sub,
    # --- Comparison (AlloyBuffer operators / buffer_ops._compare_nd) ---
    torch.ops.aten.eq.Tensor: _eq,
    torch.ops.aten.eq.Scalar: _eq,
    torch.ops.aten.ge.Tensor: AlloyBuffer.__ge__,
    torch.ops.aten.ge.Scalar: AlloyBuffer.__ge__,
    torch.ops.aten.gt.Tensor: AlloyBuffer.__gt__,
    torch.ops.aten.gt.Scalar: AlloyBuffer.__gt__,
    torch.ops.aten.le.Tensor: AlloyBuffer.__le__,
    torch.ops.aten.le.Scalar: AlloyBuffer.__le__,
    torch.ops.aten.lt.Tensor: AlloyBuffer.__lt__,
    torch.ops.aten.lt.Scalar: AlloyBuffer.__lt__,
    torch.ops.aten.ne.Tensor: _ne,
    torch.ops.aten.ne.Scalar: _ne,
    # --- Remaining handlers ---
    torch.ops.aten.clamp.default: _clamp,
    torch.ops.aten.clamp.Tensor: _clamp,
    torch.ops.aten.div.Tensor: AlloyBuffer.__truediv__,
    torch.ops.aten.div.Scalar: AlloyBuffer.__truediv__,
    torch.ops.aten.full_like.default: _full_like,
    torch.ops.aten.scatter.value: _scatter_value,
    torch.ops.aten.scatter_add.default: _scatter_add,
    torch.ops.aten.gather.default: _gather,
    torch.ops.aten.lift_fresh_copy.default: lambda x: x,
    torch.ops.aten.mean.dim: _mean_dim,
    torch.ops.aten.mean.default: lambda x, *, dtype=None: _mean_dim(x, None, dtype=dtype),
    torch.ops.aten.var.correction: _var_correction,
    torch.ops.aten.new_empty_strided.default: _new_empty_strided,
    torch.ops.aten.copy.default: _copy,
    torch.ops.aten.mm.default: lambda a, b: _mm(a, b),
    torch.ops.aten.native_group_norm.default: _native_group_norm,
    torch.ops.aten.native_layer_norm.default: _native_layer_norm,
    torch.ops.aten.ones_like.default: _ones_like,
    torch.ops.aten.permute.default: lambda x, dims: x.permute(dims),
    torch.ops.aten.pow.Tensor_Scalar: _pow_tensor_scalar,
    torch.ops.aten.pow.Scalar: _pow_scalar_tensor,
    torch.ops.aten.reciprocal.default: _reciprocal,
    torch.ops.aten.repeat.default: _repeat,
    torch.ops.aten.reshape.default: lambda x, shape: x.reshape(shape),
    torch.ops.aten.scaled_dot_product_attention.default: _scaled_dot_product_attention,
    torch.ops.aten.scalar_tensor.default: _scalar_tensor,
    torch.ops.aten.select.int: _select_int,
    torch.ops.aten.select_backward.default: _select_backward,
    torch.ops.aten.slice.Tensor: _slice_tensor,
    torch.ops.aten.slice_scatter.default: _slice_scatter,
    torch.ops.aten.split.Tensor: _split_tensor,
    torch.ops.aten.split_with_sizes.default: _split_with_sizes,
    torch.ops.aten.sum.default: lambda x: _sum_dim(x, None),
    torch.ops.aten.sum.dim_IntList: _sum_dim,
    torch.ops.aten.t.default: AlloyBuffer.transpose,
    torch.ops.aten.tanh.default: AlloyBuffer.tanh,
    torch.ops.aten.transpose.int: _transpose_dims,
    torch.ops.aten.squeeze.dims: _squeeze_dims,
    torch.ops.aten.squeeze.dim: _squeeze_dims,
    torch.ops.aten.unsqueeze.default: _unsqueeze,
    torch.ops.aten.view.default: lambda x, shape: x.reshape(shape),
    torch.ops.aten.where.self: _where_self,
    torch.ops.aten.zeros.default: _zeros,
    torch.ops.aten.zeros_like.default: lambda x, *, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None: (
        _zeros(
            x.shape,
            dtype=dtype or x._dtype.to_torch_dtype(),
            device=device,
            layout=layout,
            pin_memory=pin_memory,
        )
    ),
    torch.zeros: _zeros,
    "index_copy_": lambda target, dim, index, source: _cache_write_dim2_4d(
        target, dim, index, source, inplace=True
    ),
}


FX_CALL_HANDLERS = {
    **ATEN_TO_ALLOY,
    operator.getitem: operator.getitem,
    torch.ops.alloy.rms_norm.default: lambda x, w, eps: _fused_rms_norm(x, w, eps=eps),
    torch.ops.alloy.rope_apply.default: _fused_rope_apply,
    torch.ops.alloy.rope_table.default: _rope_table,
    torch.ops.alloy.rms_norm_rope.default: _fused_rms_norm_rope,
    torch.ops.alloy.rope_apply_backward.default: _fused_rope_apply_backward,
    torch.ops.alloy.rms_norm_backward.default: _fused_rms_norm_backward,
    torch.ops.alloy.batched_mm.default: _alloy_batched_mm_handler,
    torch.ops.alloy.gemm_residual_layernorm.default: _alloy_gemm_residual_layernorm_handler,
    torch.ops.alloy.gemm_residual_rmsnorm.default: _alloy_gemm_residual_rmsnorm_handler,
    torch.ops.alloy.dequant_mm.default: _alloy_dequant_mm_handler,
    torch.ops.alloy.batched_dequant_mm.default: _alloy_batched_dequant_mm_handler,
    torch.ops.alloy.dot_silu.default: _alloy_dot_silu_handler,
    torch.ops.alloy.dequant_silu.default: _alloy_dequant_silu_handler,
    torch.ops.alloy.gguf_q4_k_embedding.default: _alloy_gguf_q4_k_embedding_handler,
    torch.ops.alloy.gguf_q4_k_mm.default: _alloy_gguf_q4_k_mm_handler,
    torch.ops.alloy.batched_gguf_q4_k_mm.default: _alloy_batched_gguf_q4_k_mm_handler,
    torch.ops.alloy.gguf_q4_k_silu.default: _alloy_gguf_q4_k_silu_handler,
    torch.ops.alloy.gguf_q4_k_gelu.default: _alloy_gguf_q4_k_gelu_handler,
    torch.ops.alloy.mlx_q4_mm.default: _alloy_mlx_q4_mm_handler,
    torch.ops.alloy.batched_mlx_q4_mm.default: _alloy_batched_mlx_q4_mm_handler,
    torch.ops.alloy.mlx_q4_silu.default: _alloy_mlx_q4_silu_handler,
    torch.ops.alloy.mlx_q4_embedding.default: _alloy_mlx_q4_embedding_handler,
    torch.ops.alloy.gguf_moe_routed.default: _gguf_moe_routed_handler,
    torch.ops.alloy.gguf_q5_0_embedding.default: _alloy_gguf_q5_0_embedding_handler,
    torch.ops.alloy.gguf_q5_0_mm.default: _alloy_gguf_q5_0_mm_handler,
    torch.ops.alloy.batched_gguf_q5_0_mm.default: _alloy_batched_gguf_q5_0_mm_handler,
    torch.ops.alloy.gguf_q6_k_mm.default: _alloy_gguf_q6_k_mm_handler,
    torch.ops.alloy.batched_gguf_q6_k_mm.default: _alloy_batched_gguf_q6_k_mm_handler,
    torch.ops.alloy.gguf_q6_k_embedding.default: _alloy_gguf_q6_k_embedding_handler,
    torch.ops.alloy.gguf_q8_0_embedding.default: _alloy_gguf_q8_0_embedding_handler,
    torch.ops.alloy.gguf_q8_0_mm.default: _alloy_gguf_q8_0_mm_handler,
    torch.ops.alloy.batched_gguf_q8_0_mm.default: _alloy_batched_gguf_q8_0_mm_handler,
    torch.ops.alloy.gguf_q8_0_silu.default: _alloy_gguf_q8_0_silu_handler,
    torch.ops.alloy.linear_attention_update.default: _linear_attention_update_handler,
    torch.ops.alloy.short_conv_update.default: _short_conv_update_handler,
    torch.ops.alloy.short_conv_gated.default: _short_conv_gated_handler,
    torch.ops.alloy.attention_cache.default: _attention_cache_handler,
    torch.ops.alloy.attention_cache_q8.default: _attention_cache_q8_handler,
    torch.ops.alloy.attention_kv_update.default: _attention_kv_update_handler,
    torch.ops.alloy.attention_kv_update_multi.default: _attention_kv_update_multi_handler,
    torch.ops.alloy.attention_kv_update_multi_bidir.default: _attention_kv_update_multi_bidir_handler,
    torch.ops.alloy.spec_kv_write.default: _spec_kv_write_handler,
    torch.ops.alloy.attention_prefill_cold.default: _attention_prefill_cold_handler,
    torch.ops.alloy.attention_prefill_warm.default: _attention_prefill_warm_handler,
    torch.ops.alloy.cross_entropy_fwd_fused.default: _alloy_cross_entropy_fwd_fused_handler,
    torch.ops.alloy.sample_categorical.default: _sample_categorical_handler,
    torch.ops.alloy.cross_entropy_bwd_fused.default: _alloy_cross_entropy_bwd_fused_handler,
}
