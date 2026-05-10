# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.utils import WeightsMapper

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


class BaseGGUFWeightsAdapter(ABC):
    """Base hooks for GGUF weight loading adapters."""

    def __init__(self, config: "PretrainedConfig") -> None:
        self.config = config

    @classmethod
    @abstractmethod
    def matches(cls, config: "PretrainedConfig") -> bool:
        """Return whether this adapter supports *config*."""

    @abstractmethod
    def build_name_map(self, model_config: "ModelConfig") -> dict[str, str]:
        """Build the ``{gguf_name: hf_name}`` mapping for the main GGUF file."""

    def patch_hf_config(
        self,
        model_path: str,
        hf_config: "PretrainedConfig",
    ) -> "PretrainedConfig":
        """Patch HF config before model init."""
        del model_path
        return hf_config

    def extra_gguf_files(self, model_path: str) -> list[str]:
        """Return extra GGUF files to load."""
        del model_path
        return []

    def is_extra_param(self, hf_name: str) -> bool:
        """Return whether a HF parameter is handled by extra GGUF files."""
        del hf_name
        return False

    def get_weights_mapper(self) -> WeightsMapper | None:
        """Return mapper for extra GGUF files, if any."""
        return None

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Transform one loaded weight."""
        del hf_name
        return weight
