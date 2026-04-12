# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
    register_quantization_config,
)
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.config.load import LoadConfig
from vllm.model_format import register_model_format

from .loader import GGUFModelLoader
from .model_format import GGUFModelFormat
from .quantization import GGUFConfig

OOTGGUFConfig = GGUFConfig
OOTGGUFModelLoader = GGUFModelLoader


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    if "gguf" not in QUANTIZATION_METHODS or get_quantization_config("gguf") is not GGUFConfig:
        register_quantization_config("gguf")(GGUFConfig)

    if (
        "gguf" not in _LOAD_FORMAT_TO_MODEL_LOADER
        or not isinstance(get_model_loader(LoadConfig(load_format="gguf")), GGUFModelLoader)
    ):
        register_model_loader("gguf")(GGUFModelLoader)

    register_model_format(GGUFModelFormat())
