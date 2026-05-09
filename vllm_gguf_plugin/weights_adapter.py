# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model adapters for GGUF weight loading.

Each adapter can optionally provide a :class:`~vllm.model_executor.models.utils.WeightsMapper`
that renames raw GGUF tensor names (from extra files such as ``mmproj-*.gguf``)
to HF-style parameter names.  The loader then feeds those renamed weights
through the model's own ``load_weights`` / ``hf_to_vllm_mapper`` pipeline.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import regex
import torch
from transformers import AutoModelForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

from .weight_utils import gguf_quant_weights_iterator, gguf_quant_weights_iterator_multi

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


class GGUFWeightsAdapter:
    """Minimal base adapter.  Subclass and override only what you need."""

    def __init__(self, config: "PretrainedConfig") -> None:
        self.config = config

    @classmethod
    def matches(cls, config: "PretrainedConfig") -> bool:
        return True  # fallback

    @property
    def auto_model_class(self):
        return AutoModelForCausalLM

    def extra_gguf_files(self, model_path: str) -> list[str]:
        """Extra GGUF files to load (e.g. ``mmproj-*.gguf``)."""
        return []

    def is_extra_param(self, hf_name: str) -> bool:
        """Return True for params covered by :meth:`get_weights_mapper`.

        The loader skips these from the gguf-py name-map lookup and from the
        "unmapped params" error.
        """
        return False

    def get_weights_mapper(self) -> WeightsMapper | None:
        """Return a :class:`WeightsMapper` to rename raw GGUF tensor names
        from extra files to HF-style parameter names, or ``None`` to fall
        back to the shared ``gguf_to_hf_name_map``.
        """
        return None

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Apply model-specific weight conversions after name mapping."""
        return weight

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for hf_name, weight in weights:
            yield hf_name, self.transform_weight(hf_name, weight)

    def load_extra_weights(
        self,
        gguf_file: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        mapper = self.get_weights_mapper()
        if mapper is not None:
            weights = mapper.apply(gguf_quant_weights_iterator_multi([gguf_file], None))
        else:
            weights = gguf_quant_weights_iterator(gguf_file, gguf_to_hf_name_map)
        yield from self.map_weights(weights)

    def load_weights(
        self,
        gguf_files: list[str],
        gguf_to_hf_name_map: dict[str, str],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        if len(gguf_files) > 1:
            weights = gguf_quant_weights_iterator_multi(gguf_files, gguf_to_hf_name_map)
        else:
            weights = gguf_quant_weights_iterator(gguf_files[0], gguf_to_hf_name_map)
        yield from self.map_weights(weights)

    def build_name_map(self, model_config: "ModelConfig") -> dict[str, str]:
        """Build the ``{gguf_name: hf_name}`` mapping for the main GGUF file.

        Subclasses may override, but the default implementation uses gguf-py's
        :func:`gguf.get_tensor_name_map` plus a dummy model state-dict to
        derive the full mapping automatically.
        """
        from vllm.logger import init_logger
        logger = init_logger(__name__)

        config = model_config.hf_config
        text_config = config.get_text_config()
        model_type = config.model_type
        is_multimodal = (
            hasattr(config, "vision_config") and config.vision_config is not None
        )

        gguf_to_hf_name_map: dict[str, str] = {}
        sideload_params: list[re.Pattern] = []

        # gguf architecture name normalisations
        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            model_type = "gemma3"
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type == "minimax_m2":
            model_type = "minimax-m2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.block_sparse_moe.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w3.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.block_sparse_moe\.experts\.(gate_up_proj|down_proj)"
                    )
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        text_name_map = gguf.get_tensor_name_map(arch, text_config.num_hidden_layers)

        # Build vision name map only when the adapter has no WeightsMapper
        # (i.e. no extra file handling for vision params).
        if is_multimodal and self.get_weights_mapper() is None:
            mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
            vision_name_map = gguf.get_tensor_name_map(
                mm_proj_arch, config.vision_config.num_hidden_layers
            )
        else:
            vision_name_map = None

        with torch.device("meta"):
            dummy_model = self.auto_model_class.from_config(
                config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):
            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        if model_type == "minimax-m2" and not hf_checkpoint_map:
            state_dict = {
                name.replace(".mlp.", ".block_sparse_moe."): tensor
                for name, tensor in state_dict.items()
            }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if is_multimodal and hf_name.startswith("model."):
                hf_name = hf_name[6:]
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]
                if is_multimodal:
                    hf_name = "model." + hf_name
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                base_name, suffix = hf_name, ""
                if base_name.endswith("_weight"):
                    base_name = base_name[:-7]
                    suffix = "weight"
            gguf_name = None
            if vision_name_map is not None:
                gguf_name = vision_name_map.get_name(base_name)
            if gguf_name is None:
                gguf_name = text_name_map.get_name(base_name)
            if gguf_name is None:
                return None
            return gguf_name + "." + suffix

        unmapped_params = []
        for hf_name in state_dict:
            if self.is_extra_param(hf_name):
                continue
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                unmapped_params.append(hf_name)

        if unmapped_params:
            unmapped_params = [
                x for x in unmapped_params
                if not any(regex.fullmatch(p, x) for p in sideload_params)
            ]
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map GGUF parameters "
                f"({len(unmapped_params)}): {unmapped_params}"
            )
        return gguf_to_hf_name_map


# ---------------------------------------------------------------------------
# Gemma3 multimodal adapter
# ---------------------------------------------------------------------------

_VT = "model.vision_tower.vision_model."   # → vision_tower. after hf_to_vllm_mapper
_MM = "model.multi_modal_projector."       # → multi_modal_projector. after mapper

_BLK_COMP: dict[str, str] = {
    "ln1":      "layer_norm1",
    "ln2":      "layer_norm2",
    "attn_q":   "self_attn.q_proj",
    "attn_k":   "self_attn.k_proj",
    "attn_v":   "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ffn_down": "mlp.fc1",   # GGUF ffn_down = expansion = fc1
    "ffn_up":   "mlp.fc2",   # GGUF ffn_up   = contraction = fc2
}


def _build_gemma3_mapper() -> WeightsMapper:
    orig_to_new_regex: dict[re.Pattern, str] = {
        re.compile(rf"^v\.blk\.(\d+)\.{gguf_comp}\.(weight|bias)$"):
            rf"{_VT}encoder.layers.\1.{vllm_comp}.\2"
        for gguf_comp, vllm_comp in _BLK_COMP.items()
    }
    orig_to_new_prefix: dict[str, str] = {
        "mm.input_projection.weight":  f"{_MM}mm_input_projection_weight",
        "mm.soft_emb_norm.weight":     f"{_MM}mm_soft_emb_norm.weight",
        "v.patch_embd.weight":    f"{_VT}embeddings.patch_embedding.weight",
        "v.patch_embd.bias":      f"{_VT}embeddings.patch_embedding.bias",
        "v.position_embd.weight": f"{_VT}embeddings.position_embedding.weight",
        "v.post_ln.weight":       f"{_VT}post_layernorm.weight",
        "v.post_ln.bias":         f"{_VT}post_layernorm.bias",
    }
    return WeightsMapper(
        orig_to_new_regex=orig_to_new_regex,
        orig_to_new_prefix=orig_to_new_prefix,
    )


class Gemma3GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for Gemma3 GGUF models."""

    _EXTRA_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")
    _mapper: WeightsMapper | None = None
    _TEXT_PREFIX_RE = r"(?:model\.language_model\.)?model\."
    _RMS_NORM_PATTERNS = (
        regex.compile(rf"^{_TEXT_PREFIX_RE}norm\.weight$"),
        regex.compile(
            rf"^{_TEXT_PREFIX_RE}layers\.\d+\.(input|post_attention)_layernorm\.weight$"
        ),
        regex.compile(
            rf"^{_TEXT_PREFIX_RE}layers\.\d+\.(pre|post)_feedforward_layernorm\.weight$"
        ),
        regex.compile(rf"^{_TEXT_PREFIX_RE}layers\.\d+\.self_attn\.[qk]_norm\.weight$"),
        regex.compile(r"^model\.multi_modal_projector\.mm_soft_emb_norm\.weight$"),
    )
    _TEXT_GLOBAL_TENSORS = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    _TEXT_BLOCK_TENSORS = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_norm.weight": "pre_feedforward_layernorm.weight",
        "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }

    @classmethod
    def matches(cls, config: "PretrainedConfig") -> bool:
        return getattr(config, "model_type", None) in ("gemma3", "gemma3_text")

    def build_name_map(self, model_config: "ModelConfig") -> dict[str, str]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        text_prefix = ""
        if getattr(config, "vision_config", None) is not None:
            text_prefix = "model.language_model."

        mapping: dict[str, str] = {}
        for gguf_name, hf_name in self._TEXT_GLOBAL_TENSORS.items():
            if hf_name == "lm_head.weight":
                mapping[gguf_name] = hf_name
            else:
                mapping[gguf_name] = text_prefix + hf_name

        for idx in range(text_config.num_hidden_layers):
            for gguf_suffix, hf_suffix in self._TEXT_BLOCK_TENSORS.items():
                mapping[f"blk.{idx}.{gguf_suffix}"] = (
                    f"{text_prefix}model.layers.{idx}.{hf_suffix}"
                )

        return mapping

    def is_extra_param(self, hf_name: str) -> bool:
        return hf_name.startswith(self._EXTRA_PREFIXES)

    def extra_gguf_files(self, model_path: str) -> list[str]:
        from .gguf_utils import detect_gguf_multimodal
        mmproj = detect_gguf_multimodal(model_path)
        return [mmproj] if mmproj else []

    def get_weights_mapper(self) -> WeightsMapper:
        if self._mapper is None:
            self._mapper = _build_gemma3_mapper()
        return self._mapper

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if any(regex.fullmatch(p, hf_name) for p in self._RMS_NORM_PATTERNS):
            return weight - 1
        return weight


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: list[type[GGUFWeightsAdapter]] = [
    Gemma3GGUFAdapter,
]


def get_weights_adapter(config: "PretrainedConfig") -> GGUFWeightsAdapter:
    """Return the adapter for *config*, falling back to the default."""
    for cls in _ADAPTER_REGISTRY:
        if cls.matches(config):
            return cls(config)
    return GGUFWeightsAdapter(config)
