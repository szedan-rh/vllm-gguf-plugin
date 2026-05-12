# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from threading import Lock

import torch
from torch.utils import cpp_extension


_GGUF_LIBRARY_NAMESPACE = "_C_gguf"
_JIT_EXTENSION_NAME = "vllm_gguf_plugin_gguf"
_BUILD_LOCK = Lock()


def _gguf_ops_available() -> bool:
    return hasattr(torch.ops, _GGUF_LIBRARY_NAMESPACE) and hasattr(
        torch.ops._C_gguf, "ggml_dequantize"
    )


def _csrc_root() -> Path:
    return Path(__file__).resolve().parent / "csrc"


def _extension_sources() -> list[str]:
    root = _csrc_root()
    return [
        str(root / "torch_bindings.cpp"),
        str(root / "gguf" / "gguf_kernel.cu"),
    ]


def _include_paths() -> list[str]:
    root = _csrc_root()
    return [str(root), str(root / "gguf")]


def ensure_gguf_cuda_ops_loaded() -> None:
    if _gguf_ops_available():
        return

    if not torch.cuda.is_available():
        raise RuntimeError("vllm-gguf-plugin CUDA kernels require an available CUDA device.")
    if torch.version.cuda is None:
        raise RuntimeError("vllm-gguf-plugin CUDA kernels require a CUDA-enabled PyTorch build.")
    if cpp_extension.CUDA_HOME is None:
        raise RuntimeError(
            "vllm-gguf-plugin could not find the CUDA toolkit. Set CUDA_HOME before using GGUF CUDA ops."
        )

    with _BUILD_LOCK:
        if _gguf_ops_available():
            return

        build_directory = os.environ.get("VLLM_GGUF_PLUGIN_JIT_BUILD_DIR")
        if build_directory:
            Path(build_directory).mkdir(parents=True, exist_ok=True)

        cpp_extension.load(
            name=_JIT_EXTENSION_NAME,
            sources=_extension_sources(),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
            extra_include_paths=_include_paths(),
            build_directory=build_directory,
            verbose=os.environ.get("VLLM_GGUF_PLUGIN_JIT_VERBOSE") == "1",
            with_cuda=True,
        )
