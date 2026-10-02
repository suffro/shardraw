"""Phase 3, step 3: a mixture-of-experts model served by the general storage backend (decision 0006).

    uv run python benchmarks/moe_runtime.py --output experiments/phase3/<name> [--num-prompts N]

The model, prompts, storage options and configurations are in configs/phase3-moe.yaml. The
resident model runs first, on every prompt: greedy decoding with the KV cache, and every
step's logits hashed. Then its experts are released from device memory and each
configuration serves them from the drive through the same API as the Phase 1C LM head (page
store, streamer, page cache, materialization backend), through the model-agnostic adapter
`awpmi.models.moe`. Every step must reproduce the resident step bit for bit.

Writes, into the output directory:
  config.yaml, environment.json   as every benchmark
  pack.json          the expert pack's manifest (segments read from the published checkpoint),
                     expert sizes, and the device memory of each configuration
  prompts.jsonl      the prompts as this model's token ids
  reference.jsonl    per prompt and step: token, logits sha256, top-2 gap, routed experts per layer
  records.jsonl.gz   per configuration, prompt and step: routing, requested bytes, cache work,
                     bytes read from the drive, host-to-device bytes, reads, and the audit
  digest.json        sha256 of pack, prompts, reference and records (timings and memory excluded)
It stops at the first hard failure (unless --keep-going) and saves it to failure.json: a step
that differs from the resident model, or an audit that does not hold.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from awpmi.config import load_config  # noqa: E402
from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.materialization.weights import ExpertStore, WeightStore  # noqa: E402
from awpmi.models.moe import RoutingRecord, StreamedExperts, find_expert_modules, groups_from_pack, record_routing, write_expert_pack  # noqa: E402
from awpmi.storage.cache import POLICIES, PageCache  # noqa: E402
from awpmi.storage.fileio import process_memory  # noqa: E402
from awpmi.storage.pack import MANIFEST, SourceFile, open_pack  # noqa: E402
from awpmi.storage.store import IO_BLOCK_BYTES  # noqa: E402
from awpmi.streaming.streamer import PageStreamer  # noqa: E402
from awpmi.tracing import JsonlWriter, canonical_digest, environment_metadata, read_jsonl  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from storage_runtime import describe_disk  # noqa: E402

EXCLUDED_FIELDS = ("timings_ms", "system")


class HardFailure(Exception):
    pass


def logits_sha256(logits: torch.Tensor) -> str:
    return hashlib.sha256(logits.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def load_prompts(config: dict, tokenizer) -> list[dict]:
    """The Phase 1 prompts as text (decoded with their own tokenizer), tokenized for this model."""
    source_run = REPO_ROOT / config["source_run"]
    source_config, _ = load_config(source_run / "config.yaml")
    source_tokenizer = AutoTokenizer.from_pretrained(source_config.model.repository, revision=source_config.model.revision)
    prompts = []
    for row in read_jsonl(source_run / "prompts.jsonl")[: int(config["num_prompts"])]:
        text = source_tokenizer.decode(row["token_ids"])
        ids = tokenizer(text)["input_ids"][: int(config["max_prompt_tokens"])]
        prompts.append({"prompt_id": row["prompt_id"], "token_ids": ids})
    return prompts


@torch.inference_mode()
def decode(model, token_ids: list[int], steps: int, device: torch.device, before, after) -> None:
    """Greedy decoding with the KV cache; `before(step)` and `after(step, logits, positions)` frame every forward."""
    input_ids = torch.tensor([token_ids], device=device)
    cache = None
    for step in range(steps + 1):
        before(step)
        output = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        logits = output.logits[0, -1]
        after(step, logits, input_ids.shape[1])
        cache = output.past_key_values
        input_ids = logits.argmax().view(1, 1)


def audit(report: dict, requested: int, cuda: bool) -> list[str]:
    problems = []
    served, storage, transfer = report["materialization"], report["storage"], report.get("transfer", {})
    if served["requested_bytes"] != requested:
        problems.append("requested bytes != routed experts")
    if served["cache_hit_bytes"] + served["fetched_bytes"] != served["requested_bytes"]:
        problems.append("cache hits + fetched != requested")
    if storage["logical_bytes"] != served["fetched_bytes"]:
        problems.append("storage logical != fetched")
    if cuda and transfer.get("h2d_bytes") != storage["logical_bytes"]:
        problems.append("h2d != storage logical")
    if not 0 <= storage["blocks_4k"] * IO_BLOCK_BYTES - storage["physical_bytes"] < IO_BLOCK_BYTES * max(1, storage["requests"]):
        problems.append("physical bytes are not the requested rows' 4 KiB blocks")
    if storage["os_read_bytes"] is not None and (
        storage["os_read_bytes"] != storage["physical_bytes"] or storage["os_read_calls"] != storage["read_calls"]
    ):
        problems.append("OS counters differ from the store's reads")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase3-moe.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="use only the first N prompts")
    parser.add_argument("--keep-going", action="store_true", help="record hard failures instead of stopping")
    args = parser.parse_args()

    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config, prompt_config, storage = raw_config["model"], raw_config["prompts"], raw_config["storage"]
    if args.num_prompts is not None:
        prompt_config = {**prompt_config, "num_prompts": args.num_prompts}
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_config["repository"], revision=model_config["revision"])
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[model_config["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(model_config["repository"], revision=model_config["revision"], dtype=dtype)
    model = model.to(device).eval()
    modules = find_expert_modules(model)
    prompts = load_prompts(prompt_config, tokenizer)
    steps = int(prompt_config["decode_steps"])

    pack_dir = REPO_ROOT / raw_config["pack"]["directory"]
    started = time.perf_counter()
    built = not (pack_dir / MANIFEST).exists()
    if built:
        source = SourceFile(model_config["repository"], model_config["revision"], model_config["source_file"])
        write_expert_pack(model, pack_dir, sources={"checkpoint": (source, None)}, packing={"tool": "benchmarks/moe_runtime.py"})
    pack = open_pack(pack_dir, verify="segments")
    groups = groups_from_pack(pack)
    expert_bytes = sum(pack.segments[s].nbytes for g in groups.values() for s in g.segments.values())
    row_bytes = {key: sum(pack.segments[s].row_bytes for s in g.segments.values()) for key, g in groups.items()}
    pack_info = {
        "directory": raw_config["pack"]["directory"],
        "built_now": built,
        "open_and_verify_s": time.perf_counter() - started,
        "manifest": pack.manifest,
        "expert_modules": len(modules),
        "experts_per_module": sorted({g.experts for g in groups.values()}),
        "expert_bytes_total": expert_bytes,
        "expert_bytes_per_expert": sorted(set(row_bytes.values())),
        "copied_segments": sorted(name for name in pack.segments if pack.segments[name].file != "checkpoint"),
        "experts_implementation": model.config._experts_implementation,
        "configurations": {},
    }

    with open(output / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    environment = environment_metadata(
        REPO_ROOT,
        {"repository": model_config["repository"], "revision": model_config["revision"], "dtype": str(dtype), "device": str(device)},
        NUMERICS_FLAGS,
    )
    environment["num_prompts"] = len(prompts)
    environment["storage"] = {"pack_files": {k: str(p) for k, p in pack.files.items()}, "disk": describe_disk(pack.files["checkpoint"])}
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    with JsonlWriter(output / "prompts.jsonl") as log:
        for prompt in prompts:
            log.write(prompt)

    failures: list[dict] = []

    def fail(kind: str, entry: dict) -> None:
        failures.append({"kind": kind, **entry})
        if not args.keep_going:
            raise HardFailure

    routes: list[RoutingRecord] = []
    reference: dict[tuple[int, int], dict] = {}
    gc.collect()
    time.sleep(float(raw_config["timing"]["settle_seconds"]))
    handles = record_routing(model, routes.append)
    try:
        for prompt in prompts[: int(raw_config["timing"]["warmup_prompts"])]:
            decode(model, prompt["token_ids"], 1, device, lambda step: None, lambda step, logits, positions: None)
        with JsonlWriter(output / "reference.jsonl") as log:
            for prompt in prompts:

                def before(step):
                    routes.clear()

                def after(step, logits, positions, prompt=prompt):
                    top2 = torch.topk(logits.float(), 2).values.tolist()
                    entry = {
                        "prompt_id": prompt["prompt_id"],
                        "step": step,
                        "positions": positions,
                        "token": int(logits.argmax()),
                        "logits_sha256": logits_sha256(logits),
                        "top2_gap": top2[0] - top2[1],
                        "routed_per_layer": [len(route.experts) for route in routes],
                        "routed": [route.experts for route in routes],
                    }
                    reference[(prompt["prompt_id"], step)] = entry
                    log.write(entry)

                decode(model, prompt["token_ids"], steps, device, before, after)
    finally:
        for handle in handles:
            handle.remove()
    resident_device_bytes = torch.cuda.memory_allocated(device) if cuda else 0

    streamed = None
    started = time.perf_counter()
    with JsonlWriter(output / "records.jsonl.gz") as log:
        try:
            for index, entry in enumerate(raw_config["configurations"]):
                name = entry["name"]
                capacity = int(float(entry.get("cache_fraction", 0.0)) * expert_bytes)
                cache = None
                if capacity:
                    policy_class = POLICIES[entry["policy"]]
                    policy = policy_class(float(raw_config["hotness_half_life"])) if entry["policy"] == "hotness" else policy_class()
                    cache = PageCache(capacity, policy)
                backend = MaterializationBackend(
                    pack.store(
                        direct=True,
                        alignment=int(storage["alignment"]),
                        max_gap=int(storage["max_gap"]),
                        workers=int(storage["workers"]),
                        max_read_bytes=int(storage["max_read_bytes"]),
                        max_extent_bytes=int(storage["max_extent_bytes"]),
                    ),
                    device,
                    PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"])),
                    cache,
                )
                experts = ExpertStore(WeightStore(backend), groups)
                if streamed is not None:
                    streamed.remove()
                streamed = StreamedExperts(model, experts, on_route=routes.append, all_experts=bool(entry.get("all_experts", False))).install()
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
                pack_info["configurations"][name] = {
                    "cache_capacity_bytes": capacity,
                    "device_bytes_at_start": torch.cuda.memory_allocated(device) if cuda else 0,
                    "slot_buffer_bytes": streamed.buffer_bytes,
                    "host_bytes": backend.host_resident_bytes,
                }
                count = int(entry.get("num_prompts", len(prompts)))
                config_steps = int(entry.get("decode_steps", steps))
                for position, prompt in enumerate(prompts[:count]):
                    streamed.poison = index == 0 and position < int(raw_config["poison_prompts"])
                    timing = {}

                    def before(step):
                        routes.clear()
                        backend.reset_stats()
                        if cuda:
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        timing["start"] = time.perf_counter()

                    def after(step, logits, positions, prompt=prompt, name=name):
                        if cuda:
                            torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter() - timing["start"]) * 1e3
                        report = backend.report()
                        expected = reference[(prompt["prompt_id"], step)]
                        routed = [route.experts for route in routes]
                        served_layers = [len(r) if not streamed.all_experts else groups[route.module].experts for r, route in zip(routed, routes)]
                        requested = sum(n * row_bytes[route.module] for n, route in zip(served_layers, routes))
                        problems = audit(report, requested, cuda)
                        storage_stats, served = report["storage"], report["materialization"]
                        record = {
                            "configuration": name,
                            "prompt_id": prompt["prompt_id"],
                            "step": step,
                            "positions": positions,
                            "poisoned": streamed.poison,
                            "token": int(logits.argmax()),
                            "logits_sha256": logits_sha256(logits),
                            "matches_reference": logits_sha256(logits) == expected["logits_sha256"],
                            "routing_matches_reference": routed == expected["routed"],
                            "routed_per_layer": [len(r) for r in routed],
                            "served_per_layer": served_layers,
                            "requested_bytes": served["requested_bytes"],
                            "cache_hit_bytes": served["cache_hit_bytes"],
                            "fetched_bytes": served["fetched_bytes"],
                            "device_copy_bytes": served["device_copy_bytes"],
                            "storage": {k: storage_stats[k] for k in ("requests", "rows", "logical_bytes", "physical_bytes", "read_calls", "extents", "blocks_4k")},
                            "h2d_bytes": report.get("transfer", {}).get("h2d_bytes", 0),
                            "cache": None if cache is None else {k: report["cache"][k] for k in ("hits", "misses", "hit_bytes", "miss_bytes", "evictions", "bypassed")},
                            "cache_resident_bytes": None if cache is None else cache.resident_bytes,
                            "requested_fraction": served["requested_bytes"] / expert_bytes,
                            "drive_fraction": storage_stats["physical_bytes"] / expert_bytes,
                            "audit": problems,
                            "timings_ms": {"step": elapsed, "io": storage_stats["io_ms"], "copy": report.get("transfer", {}).get("copy_ms", 0.0)},
                            "system": {
                                "os_read_calls": storage_stats["os_read_calls"],
                                "os_read_bytes": storage_stats["os_read_bytes"],
                                "peak_device_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
                                "process_resident_bytes": (process_memory() or {}).get("resident_bytes"),
                            },
                        }
                        log.write(record)
                        if not record["matches_reference"] or record["token"] != expected["token"]:
                            fail("not_bitwise", {k: record[k] for k in ("configuration", "prompt_id", "step")})
                        if problems:
                            fail("audit", {"configuration": name, "prompt_id": prompt["prompt_id"], "step": step, "problems": problems})

                    decode(model, prompt["token_ids"], config_steps, device, before, after)
                print(f"{name}: {min(count, len(prompts))} prompts, {time.perf_counter() - started:.0f}s", flush=True)
                backend.store.close()
                backend.streamer.close()
        except HardFailure:
            pass
    if streamed is not None:
        streamed.remove()
    pack_info["resident_device_bytes"] = resident_device_bytes
    (output / "pack.json").write_text(json.dumps(pack_info, indent=2), encoding="utf-8")
    if failures:
        (output / "failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    pack_digest = {k: v for k, v in pack_info.items() if k not in ("open_and_verify_s", "built_now", "configurations", "resident_device_bytes")}
    digest = {
        "pack_sha256": canonical_digest([pack_digest]),
        "prompts_sha256": canonical_digest(read_jsonl(output / "prompts.jsonl")),
        "reference_sha256": canonical_digest(read_jsonl(output / "reference.jsonl")),
        "records_sha256": canonical_digest(read_jsonl(output / "records.jsonl.gz"), EXCLUDED_FIELDS),
        "excluded_fields": list(EXCLUDED_FIELDS),
    }
    (output / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - started:.0f}s; hard failures: {len(failures)}; output: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
