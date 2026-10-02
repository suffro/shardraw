"""Phase 1C evidence for the two fallback decisions of decision 0004.

    uv run python benchmarks/fallback_study.py --output experiments/phase1c/fallback-study

1. Masked fallback (row independence). Does F.linear(h, W') reproduce bitwise the rows
   kept in W' (zeros elsewhere), in the reference's own input shape [1, 1, K]? Checked
   for LM-head shapes of several models, BF16 and FP32, CUDA and CPU, several input
   distributions (including real SmolLM2 LM-head inputs) and row subsets from one
   row to half of them, including the top rows of the logits, which is what a real
   fallback keeps.
2. Output rounding of the reference GEMM. Rows with exactly two non-zero products
   whose sum is representable in binary32 make the accumulator exact whatever the
   summation order, so the BF16 output reveals the rounding of the epilogue alone:
   round-to-nearest-even, toward zero, down, or up. Midpoints, values just around
   them, both signs and several binades are covered.

Writes masked.jsonl, rounding.jsonl, environment.json and summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from awpmi.bounds.floating import round_down_to_grid, round_up_to_grid  # noqa: E402
from awpmi.config import load_config  # noqa: E402
from awpmi.models.smollm2 import ModelSpec, final_hidden_state, lm_head_weight, load_model, resolve_dtype  # noqa: E402
from awpmi.tracing import JsonlWriter, environment_metadata, read_jsonl  # noqa: E402

# (name, vocabulary, hidden size): LM heads of public models, plus odd shapes.
SHAPES = [
    ("smollm2-135m", 49152, 576),
    ("smollm2-360m", 49152, 960),
    ("smollm2-1.7b", 49152, 2048),
    ("qwen2.5-0.5b", 151936, 896),
    ("llama-3.2-1b", 128256, 2048),
    ("llama-2-7b", 32000, 4096),
    ("odd-1000x576", 1000, 576),
    ("odd-4097x333", 4097, 333),
]
CPU_MAX_ELEMENTS = 64_000_000  # CPU GEMVs above this size take too long to repeat
DENSITIES = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5)  # 0.0: a single row
TOP_ROWS = (1, 2, 5, 20)
SCALES = (0.1, 1.0, 10.0, 100.0)


def masks(rows: int, logits: torch.Tensor, generator: torch.Generator) -> list[tuple[str, torch.Tensor]]:
    found = []
    for density in DENSITIES:
        keep = torch.rand(rows, generator=generator) < density
        keep[int(torch.randint(rows, (1,), generator=generator))] = True
        found.append((f"random-{density:g}", keep))
    order = torch.argsort(logits.float().cpu(), descending=True, stable=True)
    for count in TOP_ROWS:
        keep = torch.zeros(rows, dtype=torch.bool)
        keep[order[:count]] = True
        found.append((f"top-{count}", keep))
    return found


def masked_checks(weight: torch.Tensor, inputs: list[tuple[str, torch.Tensor]], seed: int) -> dict[tuple[str, str], list[int]]:
    """{(input kind, mask kind): [checks, mismatches]} for one weight."""
    generator = torch.Generator().manual_seed(seed)
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    zero = torch.zeros((), dtype=weight.dtype, device=weight.device)
    for kind, hidden in inputs:
        full = F.linear(hidden.view(1, 1, -1), weight).reshape(-1)
        for mask_kind, keep in masks(weight.shape[0], full, generator):
            keep = keep.to(weight.device)
            masked = F.linear(hidden.view(1, 1, -1), torch.where(keep[:, None], weight, zero)).reshape(-1)
            entry = counts[(kind, mask_kind)]
            entry[0] += 1
            entry[1] += int(not torch.equal(full[keep], masked[keep]))
    return counts


def synthetic_inputs(columns: int, dtype: torch.dtype, device: torch.device, seed: int) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    found = []
    for scale in SCALES:
        found.append((f"normal-x{scale:g}", (torch.randn(columns, generator=generator) * scale).to(dtype).to(device)))
    heavy = torch.randn(columns, generator=generator) * 10.0 ** torch.randint(-3, 3, (columns,), generator=generator)
    found.append(("heavy-tailed", heavy.to(dtype).to(device)))
    sparse = torch.zeros(columns)
    sparse[torch.randint(columns, (4,), generator=generator)] = 50.0
    found.append(("sparse", sparse.to(dtype).to(device)))
    return found


def nearest_even(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    down, up = round_down_to_grid(values, dtype), round_up_to_grid(values, dtype)
    below, above = values - down, up - values
    even = (down.to(dtype).view(torch.int16) & 1) == 0
    return torch.where(below < above, down, torch.where(above < below, up, torch.where(even, down, up)))


def rounding_cases(rows: int, columns: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """BF16 rows W[j] = (a_j, b_j, 0, …) and h = (1, 1, 0, …): exact accumulators a_j + b_j (float64)."""
    generator = torch.Generator().manual_seed(0)
    a_values, b_values, kinds = [], [], []
    while len(a_values) < rows:
        exponent = int(torch.randint(-6, 8, (1,), generator=generator))
        mantissa = int(torch.randint(128, 256, (1,), generator=generator))  # 8 significant bits
        a = mantissa * 2.0 ** (exponent - 7)  # in [2^exponent, 2^(exponent+1))
        ulp = 2.0 ** (exponent - 7)
        tiny = ulp * 2.0**-6  # far above binary32 resolution at this binade: a + b stays exact
        sign = 1.0 if bool(torch.rand(1, generator=generator) < 0.5) else -1.0
        for kind, offset in (
            ("midpoint", ulp / 2),
            ("below-midpoint", ulp / 2 - tiny),
            ("above-midpoint", ulp / 2 + tiny),
            ("small-positive", tiny),
            ("small-negative", -tiny),
            ("negative-midpoint", -ulp / 2),
        ):
            a_values.append(sign * a)
            b_values.append(sign * offset)
            kinds.append(f"{kind}{'' if sign > 0 else '/neg'}")
    weight = torch.zeros(rows, columns, dtype=torch.float64)
    weight[:, 0] = torch.tensor(a_values[:rows], dtype=torch.float64)
    weight[:, 1] = torch.tensor(b_values[:rows], dtype=torch.float64)
    exact = weight[:, 0] + weight[:, 1]
    if not torch.equal(weight.to(torch.bfloat16).to(torch.float64), weight):
        raise ValueError("probe operands must be BF16 values")
    hidden = torch.zeros(columns, dtype=torch.float64)
    hidden[:2] = 1.0
    return weight.to(torch.bfloat16), hidden.to(torch.bfloat16), exact, kinds[:rows]


def rounding_probe(rows: int, columns: int, device: torch.device) -> dict:
    weight, hidden, exact, kinds = rounding_cases(rows, columns)
    output = F.linear(hidden.to(device).view(1, 1, -1), weight.to(device)).reshape(-1).to(torch.float64).cpu()
    if not torch.equal(exact.to(torch.float32).to(torch.float64), exact):
        raise ValueError("probe accumulators must be binary32 values")
    candidates = {
        "nearest_even": nearest_even(exact, torch.bfloat16),
        "toward_zero": torch.where(exact >= 0, round_down_to_grid(exact, torch.bfloat16), round_up_to_grid(exact, torch.bfloat16)),
        "down": round_down_to_grid(exact, torch.bfloat16),
        "up": round_up_to_grid(exact, torch.bfloat16),
    }
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for index, kind in enumerate(kinds):
        by_kind[kind]["cases"] += 1
        for name, values in candidates.items():
            by_kind[kind][name] += int(output[index] == values[index])
        by_kind[kind]["faithful"] += int(output[index] in (candidates["down"][index], candidates["up"][index]))
    totals = {name: int((output == values).sum()) for name, values in candidates.items()}
    totals["faithful"] = int(((output == candidates["down"]) | (output == candidates["up"])).sum())
    return {"rows": rows, "columns": columns, "cases": len(kinds), "matches": totals, "by_kind": {k: dict(v) for k, v in by_kind.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-run", default="experiments/phase1/cs-column-baseline-run1")
    parser.add_argument("--real-prompts", type=int, default=50, help="real SmolLM2 LM-head inputs to include")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)

    devices = [torch.device("cuda")] if torch.cuda.is_available() else []
    devices.append(torch.device("cpu"))
    source_run = REPO_ROOT / args.source_run
    source_config, _ = load_config(source_run / "config.yaml")
    model_device = devices[0]
    spec = ModelSpec(
        source_config.model.repository,
        source_config.model.revision,
        resolve_dtype(source_config.model.dtype, model_device),
        model_device,
    )
    model, _ = load_model(spec)
    real_weight = lm_head_weight(model)
    prompts = read_jsonl(source_run / "prompts.jsonl")[: args.real_prompts]
    real_inputs = [
        final_hidden_state(model, torch.tensor([p["token_ids"]], device=model_device)).reshape(-1).contiguous()
        for p in prompts
    ]
    environment = environment_metadata(REPO_ROOT, {"repository": spec.repository, "revision": spec.revision}, NUMERICS_FLAGS)
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    started = time.perf_counter()
    summary = {"masked": {"checks": 0, "mismatches": 0}, "rounding": {}}
    with JsonlWriter(output / "masked.jsonl") as log:
        for device in devices:
            # The real LM head with real inputs, in the model's dtype.
            weight = real_weight.to(device)
            inputs = [(f"smollm2-input-{i}", row.to(device)) for i, row in enumerate(real_inputs)]
            inputs += synthetic_inputs(weight.shape[1], weight.dtype, device, seed=1)
            for (kind, mask_kind), (checks, mismatches) in masked_checks(weight, inputs, seed=2).items():
                log.write({"shape": "smollm2-135m-real", "dtype": str(weight.dtype), "device": device.type,
                           "input": kind, "mask": mask_kind, "checks": checks, "mismatches": mismatches})
                summary["masked"]["checks"] += checks
                summary["masked"]["mismatches"] += mismatches
            for name, rows, columns in SHAPES:
                for dtype in (torch.bfloat16, torch.float32):
                    if device.type == "cpu" and rows * columns > CPU_MAX_ELEMENTS:
                        continue
                    generator = torch.Generator().manual_seed(rows + columns)
                    weight = (torch.randn(rows, columns, generator=generator) * 0.05).to(dtype).to(device)
                    inputs = synthetic_inputs(columns, dtype, device, seed=columns)
                    for (kind, mask_kind), (checks, mismatches) in masked_checks(weight, inputs, seed=3).items():
                        log.write({"shape": name, "dtype": str(dtype), "device": device.type,
                                   "input": kind, "mask": mask_kind, "checks": checks, "mismatches": mismatches})
                        summary["masked"]["checks"] += checks
                        summary["masked"]["mismatches"] += mismatches
                    del weight
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            print(f"masked fallback on {device.type}: done, {time.perf_counter() - started:.0f}s", flush=True)

    with JsonlWriter(output / "rounding.jsonl") as log:
        for device in devices:
            result = rounding_probe(real_weight.shape[0], real_weight.shape[1], device)
            log.write({"device": device.type, **result})
            summary["rounding"][device.type] = result["matches"] | {"cases": result["cases"]}
    summary["elapsed_s"] = round(time.perf_counter() - started)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if summary["masked"]["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
