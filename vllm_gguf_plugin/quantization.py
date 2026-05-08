# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from . import ops

logger = init_logger(__name__)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_name(self) -> QuantizationMethods:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # GGUF dequantization kernels use half precision (fp16) internally.
        # bfloat16 has precision issues on Blackwell devices.
        if current_platform.has_device_capability(100):
            logger.warning_once("GGUF has precision issues with bfloat16 on Blackwell.")
            return [torch.half, torch.float32]
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None
    ) -> "QuantizationMethods | None":
        # When user explicitly specifies --quantization gguf, override
        # whatever quantization method is in the HF model config (e.g. fp8).
        if user_quant == "gguf":
            return "gguf"
        return None

    @classmethod
    def requires_hf_quant_config(cls) -> bool:
        return False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedEmbeddingMethod()
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            # TODO: Select UnquantizedFusedMoEMethod on unquantized layers.
            return GGUFMoEMethod(self, layer.moe_config)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        :param hf_to_vllm_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to vllm model structure
        """
        if self.unquantized_modules is not None:
            self.unquantized_modules = hf_to_vllm_mapper.apply_list(
                self.unquantized_modules
            )


def is_layer_skipped_gguf(
    prefix: str,
    unquantized_modules: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
):
    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                shard_prefix in module_name for module_name in unquantized_modules
            )

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(module_name in prefix for module_name in unquantized_modules)

    assert is_skipped is not None
    return is_skipped


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    # enable MMVQ in contiguous batching with batch_size=1
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf

except AttributeError as error:
    raise error


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(x: torch.Tensor):
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        apply_moe_activation(activation_enum, out, x)
        return out

    # lazy import to avoid triggering triton import in CPU backend
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    # unless we decent expert reuse we are better off running moe_vec kernel
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = ops.ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_SIZE, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ops.ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    else:
        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = fused_mul_mat_gguf(
                    out, expert_down, qweight_type2
                ).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf

except AttributeError as error:
    raise error


def _apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ops.ggml_dequantize(
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


def _apply_gguf_embedding_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.empty(x.shape[0], hidden_size, dtype=dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_apply_gguf_embedding",
        op_func=_apply_gguf_embedding,
        fake_impl=_apply_gguf_embedding_fake,
    )
    apply_gguf_embedding = torch.ops.vllm._apply_gguf_embedding

except AttributeError as error:
    raise error


def _clone_loaded_weight(loaded_weight: torch.Tensor) -> torch.Tensor:
    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)
    return loaded_weight.detach().clone()


def _resolve_gguf_weight_loader(
    layer: torch.nn.Module,
    fallback_weight_loader=None,
):
    return (
        layer.weight_loader_v2
        if hasattr(layer, "weight_loader_v2")
        else fallback_weight_loader
    )


def _resolve_gguf_weight_type_loader(
    layer: torch.nn.Module,
    fallback_weight_loader=None,
):
    """Weight loader for GGUF weight-type parameters.

    Wraps the layer's weight_loader_v2 but intercepts the
    ``loaded_shard_id=None`` (fused-on-disk, e.g. GPT-2 c_attn) case.
    For fused checkpoints all Q/K/V shards share the same quant type, so
    we bypass ``_load_fused_module_from_checkpoint`` (which requires
    ``output_dim`` and tries to ``narrow`` a scalar tensor) and simply
    store the weight type directly.
    """
    base_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    if base_loader is None:
        return fallback_weight_loader

    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):
        if loaded_shard_id is None and hasattr(param, "_store"):
            # Fused checkpoint: same quant type for all shards — store it once.
            param._store(loaded_weight)
            return
        base_loader(param, loaded_weight, loaded_shard_id)

    return _gguf_weight_type_loader_v2


def _materialize_parameter_data(
    param: Parameter | UninitializedParameter,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if isinstance(param, UninitializedParameter):
        param.materialize(shape, device=param.device, dtype=dtype)


def _gguf_shard_id_as_int(shard_id: int | str) -> int:
    if isinstance(shard_id, int):
        return shard_id
    qkv_idxs = {"q": 0, "k": 1, "v": 2}
    return qkv_idxs[shard_id]


def _gguf_ordered_shard_ids(shard_ids: list[int | str]) -> list[int | str]:
    return sorted(shard_ids, key=_gguf_shard_id_as_int)


def _store_gguf_loaded_weight(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    loaded_weight = _clone_loaded_weight(loaded_weight).to(device=param.device)
    if shard_id is None:
        _materialize_parameter_data(param, tuple(loaded_weight.shape), loaded_weight.dtype)
        param.data.copy_(loaded_weight)
        return

    if shard_id not in param.shard_id_map:
        param.shard_id_map[shard_id] = len(param.data_container)
        param.data_container.append(loaded_weight)
        param.shard_id.append(shard_id)
    else:
        param.data_container[param.shard_id_map[shard_id]] = loaded_weight
    if not isinstance(param, UninitializedParameter) and param.data.numel() == 0:
        param.data = loaded_weight


def _store_gguf_weight_type(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    loaded_weight = _clone_loaded_weight(loaded_weight).to(
        device=param.device, dtype=torch.uint8
    )
    weight_type = int(loaded_weight.item())
    num_elements = getattr(param, "num_elements", 1)
    if shard_id is None:
        _materialize_parameter_data(param, (num_elements,), torch.uint8)
        param.weight_type = weight_type
        if param.data.numel() == 1:
            param.data.fill_(weight_type)
        else:
            param.data.zero_()
            param.data[0] = weight_type
        return

    param.shard_weight_type[shard_id] = weight_type
    if len(param.shard_weight_type) == 1:
        param.weight_type = weight_type
    if not isinstance(param, UninitializedParameter):
        if param.data.numel() == 0:
            param.data = torch.empty(
                num_elements, dtype=torch.uint8, device=loaded_weight.device
            )
        param.data[_gguf_shard_id_as_int(shard_id)] = weight_type


def _gguf_embedding_weight_loader(
    layer: VocabParallelEmbedding,
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
) -> None:
    loaded_weight = _clone_loaded_weight(loaded_weight).to(device=param.device)
    start_idx = layer.shard_indices.org_vocab_start_index
    shard_size = layer.shard_indices.org_vocab_end_index - start_idx
    loaded_weight = loaded_weight.narrow(param.output_dim, start_idx, shard_size)

    padded_shape = list(loaded_weight.shape)
    padded_shape[param.output_dim] = param.tensor_shape[param.output_dim]
    _materialize_parameter_data(param, tuple(padded_shape), loaded_weight.dtype)
    param.data.zero_()
    param.data.narrow(param.output_dim, 0, loaded_weight.shape[param.output_dim]).copy_(
        loaded_weight
    )


def _gguf_embedding_weight_type_loader(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
) -> None:
    _store_gguf_weight_type(param, loaded_weight)


class _GGUFParamLoadMixin:
    """Mixin providing GGUF parameter weight loading methods.

    TP slicing is applied here because vllm's weight_loader_v2 calls these
    methods with the full loaded tensor, expecting the parameter itself to
    apply the correct per-rank slice. GGUF tensors are always quantized
    row-wise, so output-parallel layers (column/merged/qkv) slice along
    dim 0 (output rows), while row-parallel layers slice along dim 1
    (encoded input columns).
    """

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and loaded_weight.ndim >= 1:
            shard_size = loaded_weight.shape[0] // tp_size
            if shard_size > 0:
                loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)
        self._store(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        # Row-parallel slices along the input dimension (encoded columns, dim 1).
        # GGUF quantizes in blocks along rows, so column-slicing is valid when
        # input_size is divisible by (block_size * tp_size).
        if tp_size > 1 and loaded_weight.ndim >= 2:
            shard_size = loaded_weight.shape[1] // tp_size
            if shard_size > 0:
                loaded_weight = loaded_weight.narrow(1, tp_rank * shard_size, shard_size)
        self._store(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        tp_rank = kwargs.get("tp_rank", 0)
        shard_size = kwargs.get("shard_size")  # per-rank output rows
        if shard_size is not None and loaded_weight.ndim >= 1 and shard_size > 0:
            if shard_size < loaded_weight.shape[0]:
                loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)
        self._store(loaded_weight, shard_id=shard_id)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        tp_rank = kwargs.get("tp_rank", 0)
        shard_size = kwargs.get("shard_size")  # per-rank output rows
        # num_heads here is num_kv_head_replicas (for k/v in GQA models)
        num_kv_head_replicas = kwargs.get("num_heads", 1)
        if shard_size is not None and loaded_weight.ndim >= 1 and shard_size > 0:
            if shard_size < loaded_weight.shape[0]:
                # For k/v with GQA replicas, multiple ranks share the same heads
                effective_tp_rank = (
                    tp_rank // num_kv_head_replicas
                    if shard_id in ("k", "v")
                    else tp_rank
                )
                loaded_weight = loaded_weight.narrow(
                    0, effective_tp_rank * shard_size, shard_size
                )
        self._store(loaded_weight, shard_id=shard_id)


class GGUFWeightParameter(_GGUFParamLoadMixin, BasevLLMParameter):
    def __init__(
        self,
        *,
        data: torch.Tensor,
        weight_loader,
        input_dim: int,
        output_dim: int,
        tensor_shape: tuple[int, ...],
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.tensor_shape = tensor_shape
        self.data_container: list[torch.Tensor] = []
        self.shard_id: list[int | str] = []
        self.shard_id_map: dict[int | str, int] = {}
        super().__init__(data=data, weight_loader=weight_loader)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _store(
        self,
        loaded_weight: torch.Tensor,
        shard_id: int | str | None = None,
    ) -> None:
        _store_gguf_loaded_weight(self, loaded_weight, shard_id)


class GGUFWeightTypeParameter(_GGUFParamLoadMixin, BasevLLMParameter):
    def __init__(self, *, data: torch.Tensor, weight_loader):
        self.weight_type = 0
        self.shard_weight_type: dict[int | str, int] = {}
        self.num_elements = data.numel()
        super().__init__(data=data, weight_loader=weight_loader)

    def _store(
        self,
        loaded_weight: torch.Tensor,
        shard_id: int | str | None = None,
    ) -> None:
        _store_gguf_weight_type(self, loaded_weight, shard_id)


def _materialize_gguf_weight_parameter(
    layer: torch.nn.Module,
    param_name: str,
    fallback_weight_loader=None,
) -> None:
    raw_param = getattr(layer, param_name)
    if isinstance(raw_param, GGUFWeightParameter):
        return

    if fallback_weight_loader is None:
        fallback_weight_loader = getattr(raw_param, "weight_loader", None)
    weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    assert weight_loader is not None
    if isinstance(raw_param, UninitializedParameter):
        data = torch.empty(0, dtype=torch.uint8, device=raw_param.device)
    else:
        data = raw_param.data
    qweight = GGUFWeightParameter(
        data=data,
        weight_loader=weight_loader,
        input_dim=raw_param.input_dim,
        output_dim=raw_param.output_dim,
        tensor_shape=raw_param.tensor_shape,
    )
    qweight.data_container = list(raw_param.data_container)
    qweight.shard_id = list(raw_param.shard_id)
    qweight.shard_id_map = dict(raw_param.shard_id_map)
    if hasattr(raw_param, "ignore_warning"):
        qweight.ignore_warning = raw_param.ignore_warning
    layer.register_parameter(param_name, qweight)


def _materialize_gguf_weight_type_parameter(
    layer: torch.nn.Module,
    param_name: str,
    fallback_weight_loader=None,
) -> None:
    raw_param = getattr(layer, param_name)
    if isinstance(raw_param, GGUFWeightTypeParameter):
        return

    if fallback_weight_loader is None:
        fallback_weight_loader = getattr(raw_param, "weight_loader", None)
    weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    assert weight_loader is not None
    num_elements = getattr(raw_param, "num_elements", 1)
    if isinstance(raw_param, UninitializedParameter):
        data = torch.empty(num_elements, dtype=torch.uint8, device=raw_param.device)
    else:
        data = raw_param.data
    qweight_type = GGUFWeightTypeParameter(data=data, weight_loader=weight_loader)
    qweight_type.num_elements = num_elements
    qweight_type.weight_type = raw_param.weight_type
    qweight_type.shard_weight_type = dict(raw_param.shard_weight_type)
    if hasattr(raw_param, "ignore_warning"):
        qweight_type.ignore_warning = raw_param.ignore_warning
    layer.register_parameter(param_name, qweight_type)

@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "needs_custom_weight_materialization": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": weight_loader_type,
                "needs_custom_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_qweight(layer)
        self._materialize_qweight_type(layer)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "qweight")

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "qweight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            # concat dim0 and pad dim1
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            # Pad the quantized weights to dense tensor, and create a map
            # with the location of each shard in the padded tensor.
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            # (dim0_start, dim0_end, dim1_size)
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            current_offset = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=qweight.weight_loader,
                input_dim=qweight.input_dim,
                output_dim=qweight.output_dim,
                tensor_shape=qweight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(qweight.shard_id_map)
            if hasattr(qweight, "ignore_warning"):
                padded_param.ignore_warning = qweight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            qweight.data_container.clear()
            qweight.shard_id.clear()
            qweight.shard_id_map.clear()
            if qweight.data.numel() > 0:
                qweight.data = torch.empty(0, dtype=qweight.dtype, device=qweight.device)
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_id = layer.qweight.shard_id

        if shard_id:
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight
            # Fall back to the global weight_type when shard_weight_type was
            # not populated (e.g. fused-on-disk checkpoints like GPT-2 c_attn
            # where all Q/K/V share the same quant type).
            fallback_wtype = layer.qweight_type.weight_type
            shard_weight_types = [
                layer.qweight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = fused_mul_mat_gguf(x, qweight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type.get(idx, fallback_wtype)
                result.append(
                    fused_mul_mat_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(
        self,
        quant_config: GGUFConfig,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # gate up proj
        w13_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "needs_custom_weight_materialization": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"needs_custom_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        # gate down proj
        w2_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "needs_custom_weight_materialization": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"needs_custom_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        return fused_moe_gguf(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            layer.w13_qweight_type.weight_type,
            layer.w2_qweight_type.weight_type,
            layer.activation.value,
        )


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        extra_weight_attrs.pop("weight_loader", None)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": partial(_gguf_embedding_weight_loader, layer),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "needs_custom_weight_materialization": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": _gguf_embedding_weight_type_loader,
                "needs_custom_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(
            layer,
            "qweight",
            fallback_weight_loader=partial(_gguf_embedding_weight_loader, layer),
        )

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(
            layer,
            "qweight_type",
            fallback_weight_loader=_gguf_embedding_weight_type_loader,
        )

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]

        return apply_gguf_embedding(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: "VocabParallelEmbedding"):
        return embed_tokens

class GGUFUninitializedParameter(_GGUFParamLoadMixin, UninitializedParameter):
    """Base class for uninitialized GGUF parameters."""
    cls_to_become = Parameter


class GGUFUninitializedWeightParameter(GGUFUninitializedParameter):
    data_container: list[torch.Tensor]

    def _store(self, loaded_weight: torch.Tensor, shard_id=None):
        _store_gguf_loaded_weight(self, loaded_weight, shard_id)


class GGUFUninitializedWeightTypeParameter(GGUFUninitializedParameter):

    def _store(self, loaded_weight: torch.Tensor, shard_id=None):
        _store_gguf_weight_type(self, loaded_weight, shard_id)
