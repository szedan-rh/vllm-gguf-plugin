# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import regex as re
from collections.abc import Generator
from typing import TYPE_CHECKING, cast

import gguf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from .weights_adapter import get_weights_adapter
from .weight_utils import (
    download_gguf,
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
)
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from .quantization import GGUFConfig

logger = init_logger(__name__)


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model_weights or model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # local_dir:quant_type (e.g. /path/to/gguf-dir:Q8_0)
        if ":" in model_name_or_path:
            local_dir, quant_type = model_name_or_path.rsplit(":", 1)
            if os.path.isdir(local_dir):
                return self._resolve_local_gguf(local_dir, quant_type)
            # remote repo_id:quant_type
            return download_gguf(
                local_dir,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, <local_dir>:<quant_type>, "
            "<repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
        )

    @staticmethod
    def _resolve_local_gguf(local_dir: str, quant_type: str) -> str:
        """Find a GGUF file matching *quant_type* in a local directory."""
        import glob as glob_mod
        patterns = [
            f"*-{quant_type}.gguf",
            f"*-{quant_type}-*.gguf",
        ]
        matches: list[str] = []
        for pat in patterns:
            matches.extend(glob_mod.glob(os.path.join(local_dir, pat)))
        if not matches:
            raise ValueError(
                f"No GGUF file matching quant_type '{quant_type}' "
                f"found in {local_dir}"
            )
        matches.sort(key=lambda x: (x.count("-"), x))
        return matches[0]

    @staticmethod
    def _get_all_gguf_files(model_path: str) -> list[str]:
        """Discover all GGUF shard files from a single shard path.

        Supports variable-width shard indices by dynamically detecting
        the padding from the original filename.
        E.g. ``*-00001-of-00005.gguf`` → all 5 shards,
             ``*-01-of-15.gguf`` → all 15 shards.
        """
        match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
        if not match:
            return [model_path]
        total = int(match.group(2))
        num_digits = len(match.group(1))
        prefix = model_path[: match.start(1)]
        suffix = model_path[match.end(2) :]
        files = []
        for i in range(1, total + 1):
            shard_path = f"{prefix}{i:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
            if os.path.isfile(shard_path):
                files.append(shard_path)
        if files:
            logger.info("Discovered %d GGUF shard files", len(files))
        return files if files else [model_path]

    def _get_gguf_weights_map(self, model_config: ModelConfig) -> dict[str, str]:
        return get_weights_adapter(model_config.hf_config).build_name_map(model_config)

    def _get_gguf_weight_type(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        gguf_files = self._get_all_gguf_files(model_name_or_path)
        weight_type_map = {}
        for f in gguf_files:
            weight_type_map.update(get_gguf_weight_type_map(f, gguf_to_hf_name_map))
        adapter = get_weights_adapter(model_config.hf_config)
        for extra_file in adapter.extra_gguf_files(model_name_or_path):
            logger.info("Loading extra mm_proj weights from %s...", extra_file)
            mapper = adapter.get_weights_mapper()
            if mapper is not None:
                raw = {
                    t.name: t.tensor_type.name
                    for t in gguf.GGUFReader(extra_file).tensors
                }
                weight_type_map.update(mapper.apply_dict(raw))
            else:
                weight_type_map.update(
                    get_gguf_weight_type_map(extra_file, gguf_to_hf_name_map)
                )
        return weight_type_map

    def _get_weights_iterator(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Iterate over all GGUF weights, loading main file(s) then extra files."""
        adapter = get_weights_adapter(model_config.hf_config)
        for extra_file in adapter.extra_gguf_files(model_name_or_path):
            yield from adapter.load_extra_weights(extra_file, gguf_to_hf_name_map)

        gguf_files = self._get_all_gguf_files(model_name_or_path)
        yield from adapter.load_weights(gguf_files, gguf_to_hf_name_map)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        model.load_weights(
            self._get_weights_iterator(model_config, local_model_path, gguf_weights_map)
        )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        gguf_files = self._get_all_gguf_files(local_model_path)
        all_extra_names = []
        for f in gguf_files:
            all_extra_names.extend(get_gguf_extra_tensor_names(f, gguf_weights_map))
        # Determine tie_word_embeddings based on whether lm_head.weight exists
        # in the GGUF file. If it's missing from the file, the model uses tied
        # embeddings. If it's present with a (potentially different) quantization,
        # we must NOT tie, otherwise the second weight load would fail with a
        # shape mismatch (e.g. Q4_K embed vs Q6_K lm_head on GPT-2).
        if "lm_head.weight" in gguf_weights_map.values():
            if "lm_head.weight" in all_extra_names:
                model_config.hf_config.update({"tie_word_embeddings": True})
            else:
                model_config.hf_config.update({"tie_word_embeddings": False})

        weight_type_map = self._get_gguf_weight_type(
            model_config, local_model_path, gguf_weights_map
        )
        # filter out unquantized modules to skip
        unquant_names = [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]
        logger.debug("GGUF unquantized modules: %s", unquant_names)
        if TYPE_CHECKING:
            vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(unquant_names)

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)
            self.load_weights(model, model_config)

            process_weights_after_loading(model, model_config, target_device)
        return model
