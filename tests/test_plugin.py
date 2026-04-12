# SPDX-License-Identifier: Apache-2.0

from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_format import get_model_format_handler

from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register


def test_register_overrides_gguf_config():
    register()

    quant_config = get_quantization_config("gguf")

    assert quant_config is OOTGGUFConfig


def test_register_overrides_gguf_loader():
    register()

    model_loader = get_model_loader(LoadConfig(load_format="gguf"))

    assert isinstance(model_loader, OOTGGUFModelLoader)


def test_register_is_idempotent():
    register()
    register()

    assert get_quantization_config("gguf") is OOTGGUFConfig
    assert isinstance(get_model_loader(LoadConfig(load_format="gguf")), OOTGGUFModelLoader)
    assert get_model_format_handler("org/model:Q4_K") is not None


def test_oot_config_reuses_in_tree_behavior():
    quant_config = OOTGGUFConfig.from_config({})

    assert isinstance(quant_config, OOTGGUFConfig)
    assert quant_config.get_name() == "gguf"
    assert repr(quant_config) == "GGUFConfig()"
