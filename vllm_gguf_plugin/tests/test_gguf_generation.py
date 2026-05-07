# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests GGUF models against unquantized models generations.
Uses vllm.LLM directly with check_logprobs_close comparison.
"""

import os
import warnings
from typing import NamedTuple

import pytest
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024


class GGUFTestConfig(NamedTuple):
    original_model: str
    gguf_model_path: str  # Full path to .gguf file


MODELS = [
    GGUFTestConfig(
        original_model="Qwen/Qwen3-0.6B",
        gguf_model_path="Qwen/Qwen3-0.6B-GGUF:Q8_0",
    ),
]


def _generate_greedy_logprobs(
    model_path: str,
    prompts: list[str],
    max_tokens: int,
    num_logprobs: int,
    tokenizer_name: str | None = None,
    quantization: str | None = None,
) -> list[tuple[list[int], str, list[dict[int, float] | None]]]:
    """Generate greedy outputs with logprobs using vllm.LLM.

    Returns list of (token_ids, text, logprobs_list) tuples.
    Each logprobs element maps token_id -> logprob value.
    """
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        quantization=quantization,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=num_logprobs,
    )
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for req_output in outputs:
        sample = req_output.outputs[0]
        token_ids = list(sample.token_ids)
        text = sample.text
        logprobs_list: list[dict[int, float] | None] = []
        if sample.logprobs:
            for lp in sample.logprobs:
                if lp is not None:
                    logprobs_list.append(
                        {tok_id: info.logprob for tok_id, info in lp.items()}
                    )
                else:
                    logprobs_list.append(None)
        results.append((token_ids, text, logprobs_list))
    return results


def check_logprobs_close(
    outputs_0_lst: list[tuple[list[int], str, list]],
    outputs_1_lst: list[tuple[list[int], str, list]],
    name_0: str,
    name_1: str,
) -> None:
    """Compare logprobs of two model outputs.

    For each generated token position:
    - If tokens match, continue
    - If tokens differ, verify each model's selected token appears in
      the other model's top-k logprobs, then break (sequences diverge)
    """
    assert len(outputs_0_lst) == len(outputs_1_lst)

    for prompt_idx, (out_0, out_1) in enumerate(zip(outputs_0_lst, outputs_1_lst)):
        ids_0, text_0, lps_0 = out_0
        ids_1, text_1, lps_1 = out_1

        if lps_0 is None:
            lps_0 = [None] * len(ids_0)
        if lps_1 is None:
            lps_1 = [None] * len(ids_1)

        for idx, (tok_0, tok_1) in enumerate(zip(ids_0, ids_1)):
            if tok_0 != tok_1:
                lp_0 = lps_0[idx] if idx < len(lps_0) else None
                lp_1 = lps_1[idx] if idx < len(lps_1) else None

                fail_msg = (
                    f"Test {prompt_idx}, token {idx}:"
                    f"\nMatched tokens: {ids_0[:idx]}"
                    f"\n{name_0}: {text_0!r}  token={tok_0}  logprobs={lp_0}"
                    f"\n{name_1}: {text_1!r}  token={tok_1}  logprobs={lp_1}"
                )

                assert lp_0 is not None, fail_msg
                assert lp_1 is not None, fail_msg
                # Each model's selected token must be in the other's top-k
                assert tok_0 in lp_1, fail_msg
                assert tok_1 in lp_0, fail_msg

                warnings.warn(fail_msg, stacklevel=2)
                break  # sequences diverge from here


def check_model_outputs(
    prompts: list[str],
    model: GGUFTestConfig,
    max_tokens: int,
    num_logprobs: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model.original_model)
    if tokenizer.chat_template is not None:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    gguf_outputs = _generate_greedy_logprobs(
        model_path=model.gguf_model_path,
        prompts=prompts[:-1],
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tokenizer_name=model.original_model,
        quantization="gguf",
    )

    original_outputs = _generate_greedy_logprobs(
        model_path=model.original_model,
        prompts=prompts[:-1],
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
    )

    check_logprobs_close(
        outputs_0_lst=original_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="original",
        name_1="gguf",
    )


@pytest.mark.parametrize(
    "model",
    MODELS,
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    example_prompts: list[str],
    model: GGUFTestConfig,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    check_model_outputs(example_prompts, model, max_tokens, num_logprobs)
