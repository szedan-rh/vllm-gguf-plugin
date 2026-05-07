# SPDX-License-Identifier: Apache-2.0

import glob
import os
from collections.abc import Generator
from pathlib import Path

import gguf
import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf

logger = init_logger(__name__)


def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    allow_patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
        f"*/*-{quant_type}.gguf",
        f"*/*-{quant_type}-*.gguf",
    ]

    logger.info("Downloading GGUF weights (%s) from %s ...", quant_type, repo_id)
    folder = download_weights_from_hf(
        model_name_or_path=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )

    local_files: list[str] = []
    for pattern in allow_patterns:
        local_files.extend(glob.glob(os.path.join(folder, pattern)))

    if not local_files:
        raise ValueError(
            f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}"
        )

    local_files.sort(key=lambda x: (x.count("-"), x))
    return local_files[0]


def get_gguf_extra_tensor_names(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> list[str]:
    reader = gguf.GGUFReader(gguf_file)
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = {tensor.name for tensor in reader.tensors}
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


def get_gguf_weight_type_map(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> dict[str, str]:
    reader = gguf.GGUFReader(gguf_file)
    return {
        gguf_to_hf_name_map[tensor.name]: tensor.tensor_type.name
        for tensor in reader.tensors
        if tensor.name in gguf_to_hf_name_map
    }


def gguf_quant_weights_iterator(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    yield from gguf_quant_weights_iterator_multi([gguf_file], gguf_to_hf_name_map)


def gguf_quant_weights_iterator_multi(
    gguf_files: list[str], gguf_to_hf_name_map: dict[str, str]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    _QUANT_TYPES = ("F32", "BF16", "F16")

    for gguf_file in gguf_files:
        reader = gguf.GGUFReader(gguf_file)
        for tensor in reader.tensors:
            if tensor.name not in gguf_to_hf_name_map:
                continue
            weight_type = tensor.tensor_type
            name = gguf_to_hf_name_map[tensor.name]

            if weight_type.name not in _QUANT_TYPES:
                yield name.replace("weight", "qweight_type"), torch.tensor(weight_type)
                name = name.replace("weight", "qweight")

            weight = tensor.data
            if weight_type.name == "BF16" and weight.dtype == np.uint8:
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    weight = weight.byteswap()
                param = torch.tensor(weight).view(torch.bfloat16)
            else:
                param = torch.tensor(weight)
            yield name, param
