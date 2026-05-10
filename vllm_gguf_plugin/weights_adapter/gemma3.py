# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re

import regex
import torch
from vllm.model_executor.models.utils import WeightsMapper

from .default import GGUFWeightsAdapter

_VT = "model.vision_tower.vision_model."
_MM = "model.multi_modal_projector."

_BLK_COMP: dict[str, str] = {
    "ln1": "layer_norm1",
    "ln2": "layer_norm2",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ffn_down": "mlp.fc1",
    "ffn_up": "mlp.fc2",
}


def _build_gemma3_mapper() -> WeightsMapper:
    orig_to_new_regex: dict[re.Pattern, str] = {
        re.compile(rf"^v\.blk\.(\d+)\.{gguf_comp}\.(weight|bias)$"): (
            rf"{_VT}encoder.layers.\1.{vllm_comp}.\2"
        )
        for gguf_comp, vllm_comp in _BLK_COMP.items()
    }
    orig_to_new_prefix: dict[str, str] = {
        "mm.input_projection.weight": f"{_MM}mm_input_projection_weight",
        "mm.soft_emb_norm.weight": f"{_MM}mm_soft_emb_norm.weight",
        "v.patch_embd.weight": f"{_VT}embeddings.patch_embedding.weight",
        "v.patch_embd.bias": f"{_VT}embeddings.patch_embedding.bias",
        "v.position_embd.weight": f"{_VT}embeddings.position_embedding.weight",
        "v.post_ln.weight": f"{_VT}post_layernorm.weight",
        "v.post_ln.bias": f"{_VT}post_layernorm.bias",
    }
    return WeightsMapper(
        orig_to_new_regex=orig_to_new_regex,
        orig_to_new_prefix=orig_to_new_prefix,
    )


class Gemma3GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for Gemma3 GGUF models."""

    _EXTRA_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")
    _mapper: WeightsMapper | None = None
    _TEXT_PREFIX_RE = r"(?:model\.language_model\.|model\.)"
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
        "token_embd.weight": "embed_tokens.weight",
        "output_norm.weight": "norm.weight",
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
    def matches(cls, config) -> bool:
        return getattr(config, "model_type", None) in ("gemma3", "gemma3_text")

    def build_name_map(self, model_config) -> dict[str, str]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        is_multimodal = getattr(config, "vision_config", None) is not None
        text_prefix = "model.language_model." if is_multimodal else "model."

        mapping: dict[str, str] = {}
        for gguf_name, hf_name in self._TEXT_GLOBAL_TENSORS.items():
            if hf_name == "lm_head.weight":
                mapping[gguf_name] = hf_name
            else:
                mapping[gguf_name] = text_prefix + hf_name

        for idx in range(text_config.num_hidden_layers):
            for gguf_suffix, hf_suffix in self._TEXT_BLOCK_TENSORS.items():
                mapping[f"blk.{idx}.{gguf_suffix}"] = (
                    f"{text_prefix}layers.{idx}.{hf_suffix}"
                )

        return mapping

    def is_extra_param(self, hf_name: str) -> bool:
        return hf_name.startswith(self._EXTRA_PREFIXES)

    def extra_gguf_files(self, model_path: str) -> list[str]:
        from ..gguf_utils import detect_gguf_multimodal

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
