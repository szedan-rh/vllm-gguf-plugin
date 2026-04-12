# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from vllm.model_format import ModelFormatHandler

from .gguf_utils import (
    check_gguf_file,
    get_gguf_file_path_from_hf,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)


class GGUFModelFormat(ModelFormatHandler):
    name = "gguf"

    def matches(self, model: str | Path | None) -> bool:
        return bool(model) and is_gguf(model)

    def update_engine_args(self, engine_args: Any) -> None:
        engine_args.quantization = "gguf"
        engine_args.load_format = "gguf"

    def prepare_hf_config_load(
        self,
        model: str | Path,
        revision: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str | Path, dict[str, Any]]:
        kwargs = dict(kwargs or {})
        if check_gguf_file(model):
            kwargs["gguf_file"] = Path(model).name
            return Path(model).parent, kwargs
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id, kwargs
        return model, kwargs

    def should_use_hf_config_parser(
        self,
        original_model: str | Path,
        resolved_model: str | Path,
    ) -> bool:
        return check_gguf_file(original_model)

    def get_missing_hf_config_error(
        self,
        original_model: str | Path,
        resolved_model: str | Path,
    ) -> str | None:
        if is_remote_gguf(original_model):
            return (
                "Could not find config.json for remote GGUF model repo. "
                "To load remote GGUF model through `<repo_id>:<quant_type>`, "
                "ensure your model has config.json (HF format) file. "
                "Otherwise please specify --hf-config-path <original_repo> "
                "in engine args to fetch config from unquantized hf model."
            )
        return None

    def patch_parsed_hf_config(
        self,
        original_model: str | Path,
        config_dict: dict[str, Any],
        config: Any,
    ) -> Any:
        if config.model_type in {"qwen3_moe"} and "norm_topk_prob" not in config_dict:
            config.update({"norm_topk_prob": True})

        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})
        return config

    def patch_model_hf_config(self, original_model: str | Path, hf_config: Any) -> Any:
        return maybe_patch_hf_config_from_gguf(str(original_model), hf_config)

    def resolve_tokenizer_init(
        self,
        tokenizer_name: str | Path,
        *args: Any,
        revision: str | None = None,
        runner_type: str = "generate",
        tokenizer_mode: str = "auto",
        **kwargs: Any,
    ) -> tuple[str | Path, tuple[Any, ...], dict[str, Any]]:
        if check_gguf_file(tokenizer_name):
            kwargs["gguf_file"] = Path(tokenizer_name).name
            tokenizer_name = Path(tokenizer_name).parent
        elif is_remote_gguf(tokenizer_name):
            tokenizer_name, quant_type = split_remote_gguf(tokenizer_name)
            kwargs["gguf_file"] = get_gguf_file_path_from_hf(
                tokenizer_name,
                quant_type,
                revision=revision,
            )
        return tokenizer_name, args, kwargs

    def resolve_processor_source(
        self,
        model_config: Any,
        component: str,
    ) -> tuple[str | Path, str | None]:
        assert not is_gguf(model_config.tokenizer), (
            "For multimodal GGUF models, the original tokenizer "
            "should be used to correctly load processor."
        )
        return model_config.tokenizer, model_config.tokenizer_revision

    def validate_model_config(self, model_config: Any) -> None:
        if is_gguf(model_config.tokenizer) and model_config.is_multimodal_model:
            raise ValueError(
                "Loading a multimodal GGUF model needs to use original "
                "tokenizer. Please specify the unquantized hf model's "
                "repo name or path using the --tokenizer argument."
            )

    def resolve_sentence_transformer_source(
        self,
        model: str | Path,
        revision: str | None = None,
    ) -> str | Path:
        if is_remote_gguf(model):
            model, _ = split_remote_gguf(model)
        return model

    def resolve_image_processor_source(
        self,
        model: str | Path,
        revision: str | None = None,
    ) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            model, _ = split_remote_gguf(model)
        return model

    def should_skip_generation_config(self, model: str | Path) -> bool:
        return is_gguf(model)
