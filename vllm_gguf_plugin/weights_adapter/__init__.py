# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .base import BaseGGUFWeightsAdapter
from .default import GGUFWeightsAdapter
from .gemma3 import Gemma3GGUFAdapter

_ADAPTER_REGISTRY: list[type[GGUFWeightsAdapter]] = [
    Gemma3GGUFAdapter,
]


def get_weights_adapter(config) -> GGUFWeightsAdapter:
    """Return the adapter for *config*, falling back to the default."""
    for cls in _ADAPTER_REGISTRY:
        if cls.matches(config):
            return cls(config)
    return GGUFWeightsAdapter(config)


__all__ = [
    "BaseGGUFWeightsAdapter",
    "GGUFWeightsAdapter",
    "Gemma3GGUFAdapter",
    "get_weights_adapter",
]
