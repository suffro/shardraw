"""Deterministic real-text prompts: random-length prefixes of wikitext-2 test paragraphs."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from awpmi.config import BenchmarkConfig
from awpmi.tracing import sha256_file


@dataclass(frozen=True)
class Prompt:
    prompt_id: int
    source_line: int
    token_ids: list[int]


def load_prompts(tokenizer, config: BenchmarkConfig) -> tuple[list[Prompt], dict[str, object]]:
    """One prompt per eligible line, lines and cut points drawn with `config.seed`.

    A line is eligible if it is not a section heading and has more than
    `min_prompt_tokens` tokens; the prompt is its first t tokens with
    t ~ U[min_prompt_tokens, min(max_prompt_tokens, len - 1)].
    """
    dataset = config.dataset
    path = hf_hub_download(dataset.repository, dataset.file, repo_type="dataset", revision=dataset.revision)
    lines = pq.read_table(path).column(dataset.text_column).to_pylist()

    eligible: list[tuple[int, list[int]]] = []
    for line_index, text in enumerate(lines):
        stripped = text.strip()
        if not stripped or stripped.startswith("="):
            continue
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) > config.min_prompt_tokens:
            eligible.append((line_index, token_ids))

    rng = random.Random(config.seed)
    chosen = rng.sample(eligible, k=min(config.num_prompts, len(eligible)))
    chosen.sort(key=lambda item: item[0])
    prompts = []
    for prompt_id, (line_index, token_ids) in enumerate(chosen):
        cut = rng.randint(config.min_prompt_tokens, min(config.max_prompt_tokens, len(token_ids) - 1))
        prompts.append(Prompt(prompt_id, line_index, token_ids[:cut]))

    provenance = {
        "repository": dataset.repository,
        "revision": dataset.revision,
        "file": dataset.file,
        "file_sha256": sha256_file(path),
        "eligible_lines": len(eligible),
        "add_special_tokens": False,
    }
    return prompts, provenance
