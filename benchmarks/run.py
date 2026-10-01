"""Phase 1 benchmark: adaptive LM head vs the fully materialized reference.

    uv run python benchmarks/run.py --config configs/smollm2-135m.yaml --output experiments/phase1/<name>

Writes, into the output directory:
  config.yaml        the config as given
  environment.json   versions, hardware, numerics flags, dataset provenance
  prompts.jsonl      the exact token ids of every prompt
  validation.jsonl   per (prompt, page width): full-materialization checks
  records.jsonl      per (prompt, page width, scheduler): the AWPMI run
  digest.json        sha256 of records/validation without timing fields
and stops at the first hard failure (unless --keep-going), saving it to failure.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402

from awpmi.bounds.residual import ReferenceNumerics  # noqa: E402
from awpmi.config import load_config  # noqa: E402
from awpmi.executor import AdaptiveLMHead  # noqa: E402
from awpmi.models.smollm2 import (  # noqa: E402
    LM_HEAD_PARAMETER,
    ModelSpec,
    final_hidden_state,
    lm_head_weight,
    load_model,
    resolve_dtype,
)
from awpmi.paging.index import build_column_page_index  # noqa: E402
from awpmi.paging.source import InMemoryPageSource  # noqa: E402
from awpmi.reference import ReferenceRunner  # noqa: E402
from awpmi.schedulers import make_scheduler  # noqa: E402
from awpmi.tracing import JsonlWriter, canonical_digest, environment_metadata, read_jsonl  # noqa: E402
from prompts import load_prompts  # noqa: E402

TIMING_FIELDS = ("elapsed_ms",)


class HardFailure(Exception):
    pass


def validate_full_materialization(head: AdaptiveLMHead, hidden: torch.Tensor, reference) -> dict:
    """Fallback reproduces the reference bitwise; page-accumulated logits lie in the derived envelope."""
    fallback_logits = head.full_materialization_logits(hidden).reshape(-1)
    partial, bounder = head.full_partial_logits(hidden)
    no_pages_left = torch.zeros(len(head.index), dtype=torch.bool, device=partial.device)
    lower, upper = bounder.logit_bounds(partial, no_pages_left)
    ref64 = reference.logits.to(torch.float64)
    return {
        "fallback_bitwise_equal": bool(torch.equal(fallback_logits, reference.logits)),
        "fallback_token": int(torch.argmax(fallback_logits)),
        "envelope_violations": int(((ref64 < lower) | (ref64 > upper)).sum()),
        "max_abs_partial_minus_reference": float((partial - ref64).abs().max()),
        "max_envelope_halfwidth": float(((upper - lower) / 2).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "smollm2-135m.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="override benchmark.num_prompts")
    parser.add_argument("--keep-going", action="store_true", help="record hard failures instead of stopping")
    args = parser.parse_args()

    config, raw_config = load_config(args.config)
    if args.num_prompts is not None:
        raw_config["benchmark"]["num_prompts"] = args.num_prompts
        config = replace(config, benchmark=replace(config.benchmark, num_prompts=args.num_prompts))

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    spec = ModelSpec(
        config.model.repository, config.model.revision, resolve_dtype(config.model.dtype, device), device
    )
    model, tokenizer = load_model(spec)
    runner = ReferenceRunner(model)
    weight = lm_head_weight(model)
    numerics = ReferenceNumerics(
        output_dtype=weight.dtype,
        accumulation_unit_roundoff=config.numerics.accumulation_unit_roundoff,
        reduction_length=weight.shape[1],
    )
    heads = {}
    page_layout = {}
    for width in config.paging.page_widths:
        index = build_column_page_index(weight, LM_HEAD_PARAMETER, width)
        heads[width] = AdaptiveLMHead(index, InMemoryPageSource(weight), numerics)
        page_layout[width] = {
            "pages_total": len(index),
            "bytes_total": index.total_bytes,
            "metadata_bytes": index.bounds.nbytes,
        }
    schedulers = {name: make_scheduler(name) for name in config.schedulers}
    prompts, dataset_provenance = load_prompts(tokenizer, config.benchmark)

    with open(output / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    model_info = {
        "repository": spec.repository,
        "revision": spec.revision,
        "dtype": str(spec.dtype),
        "device": str(device),
        "attn_implementation": model.config._attn_implementation,
        "reference_numerics": {
            "output_dtype": str(numerics.output_dtype),
            "accumulation_unit_roundoff": numerics.accumulation_unit_roundoff,
            "reduction_length": numerics.reduction_length,
        },
    }
    environment = environment_metadata(REPO_ROOT, model_info, NUMERICS_FLAGS)
    environment["dataset"] = dataset_provenance
    environment["page_layout"] = {str(w): layout for w, layout in page_layout.items()}
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    failures: list[dict] = []
    started = time.perf_counter()
    with (
        JsonlWriter(output / "prompts.jsonl") as prompt_log,
        JsonlWriter(output / "validation.jsonl") as validation_log,
        JsonlWriter(output / "records.jsonl") as record_log,
    ):
        try:
            for prompt in prompts:
                prompt_log.write({"prompt_id": prompt.prompt_id, "source_line": prompt.source_line, "token_ids": prompt.token_ids})
                input_ids = torch.tensor([prompt.token_ids], device=device)
                reference = runner.next_token(input_ids)
                hidden = final_hidden_state(model, input_ids)
                top2 = torch.topk(reference.logits.to(torch.float32), 2).values.tolist()
                prefix_equal = bool(torch.equal(hidden, reference.hidden_state))

                for width, head in heads.items():
                    validation = {
                        "prompt_id": prompt.prompt_id,
                        "page_width": width,
                        "reference_token": reference.token_id,
                        "reference_top2_gap": top2[0] - top2[1],
                        "prefix_bitwise_equal": prefix_equal,
                        **validate_full_materialization(head, hidden, reference),
                    }
                    validation_log.write(validation)
                    if (
                        not prefix_equal
                        or not validation["fallback_bitwise_equal"]
                        or validation["fallback_token"] != reference.token_id
                        or validation["envelope_violations"]
                    ):
                        failures.append({"kind": "validation", **validation})
                        if not args.keep_going:
                            raise HardFailure

                    for name, scheduler in schedulers.items():
                        tick = time.perf_counter()
                        result = head.run(hidden, scheduler)
                        elapsed_ms = (time.perf_counter() - tick) * 1e3
                        record = {
                            "prompt_id": prompt.prompt_id,
                            "page_width": width,
                            "scheduler": name,
                            "reference_token": reference.token_id,
                            "awpmi_token": result.token_id,
                            "certified": result.certified,
                            "fallback": result.fallback,
                            "pages_total": result.pages_total,
                            "pages_materialized": result.pages_materialized,
                            "materialized_fraction": result.materialized_fraction,
                            "bytes_total": result.bytes_total,
                            "bytes_materialized": result.bytes_materialized,
                            "certificate_margin": result.certificate.certificate_margin,
                            "competitor": result.certificate.competitor,
                            "materialization_order": list(result.materialization_order),
                            "margin_trajectory": [step.certificate_margin for step in result.trajectory],
                            "elapsed_ms": elapsed_ms,
                        }
                        record_log.write(record)
                        if result.token_id != reference.token_id:
                            failures.append({"kind": "certified_mismatch" if result.certified else "fallback_mismatch", **record})
                            if not args.keep_going:
                                raise HardFailure
                if (prompt.prompt_id + 1) % 100 == 0:
                    print(f"{prompt.prompt_id + 1}/{len(prompts)} prompts, {time.perf_counter() - started:.0f}s", flush=True)
        except HardFailure:
            pass

    if failures:
        (output / "failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    digest = {
        "records_sha256": canonical_digest(read_jsonl(output / "records.jsonl"), TIMING_FIELDS),
        "validation_sha256": canonical_digest(read_jsonl(output / "validation.jsonl")),
        "excluded_fields": list(TIMING_FIELDS),
    }
    (output / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - started:.0f}s; hard failures: {len(failures)}; output: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
