"""Phase 4B: the streaming reference checked against from_pretrained on Moonlight itself, where it fits (decision 0008).

    uv run python benchmarks/moonlight_reference_check.py --output experiments/phase4b/reference-check [--layers 4]

The whole model fits neither this GPU nor its host memory, so the check uses Moonlight truncated to its first layers
(`num_hidden_layers` overridden: the dense layer 0 and MoE layers after it, the final norm and the LM head), which
`from_pretrained` loads resident as published. The same truncation runs through the streaming reference
(`awpmi.streaming_reference`), one experts layer materialized at a time. Each model runs alone on the device (both do not
fit together), recording with the benchmark's recorder:

  * the sha256 of every weight and buffer, experts included (the streaming reference's as loaded, layer by layer);
  * per step and layer: attention, router logits/scores/indices/weights, experts call inputs and output, every
    (token, expert) output, shared experts, MoE block, dense MLP; per step: logits, token, the whole KV cache.

Everything must be equal bit for bit. Writes check.json (verdicts, counts, timings) and both models' records.
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
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from moonlight_runtime import (  # noqa: E402  (configures the numerics first)
    DTYPES, StepRecorder, build_prompts, compare, decode, load_adapter, step_record,
)

import torch  # noqa: E402
import transformers  # noqa: E402

from awpmi.storage.fileio import process_memory  # noqa: E402
from awpmi.streaming_reference import ReferenceCall, StreamingReference, experts_modules  # noqa: E402
from awpmi.tracing import JsonlWriter  # noqa: E402


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def weights_digest(model, skip: set[str] = frozenset()) -> dict[str, list]:
    return {
        name: [str(tensor.dtype), list(tensor.shape), tensor_sha256(tensor)]
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if name not in skip and tensor is not None
    }


def observe_resident_experts(model, recorder: StepRecorder) -> list:
    """The resident model's experts calls as `ReferenceCall`s (the same observation the streaming reference makes)."""
    handles = []
    names = ("hidden_states", "top_k_index", "top_k_weights")
    for name, module in experts_modules(model).items():

        def hook(module_, args, kwargs, output, name=name):
            values = [args[i] if len(args) > i else kwargs[names[i]] for i in range(3)]
            recorder.on_call(ReferenceCall(name, module_, *values, output))

        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    return handles


def run_prompts(model, recorder: StepRecorder, prompts: list[dict], steps: int, device) -> list[dict]:
    records = []
    for prompt in prompts:
        decode(model, prompt["token_ids"], steps, device, lambda step: recorder.clear(),
               lambda step, logits, positions, cache, prompt=prompt: records.append(step_record(prompt, step, positions, logits, cache, recorder)))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase4b-moonlight.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=4, help="keep the first N layers (layer 0 is dense)")
    parser.add_argument("--prompts", type=int, nargs="+", default=[0, 3, 5, 7], help="indices into the configured prompts")
    parser.add_argument("--decode-steps", type=int, default=4)
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config = raw["model"]
    adapter = load_adapter(model_config)
    profile = adapter.REFERENCE_PROFILE
    dtype = DTYPES[model_config["dtype"]]
    device = torch.device("cuda", torch.cuda.current_device())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config["repository"], revision=model_config["tokenizer_revision"], trust_remote_code=bool(model_config["trust_remote_code_tokenizer"])
    )
    all_prompts, provenance = build_prompts(raw["prompts"], tokenizer)
    prompts = [all_prompts[i] for i in args.prompts]
    overrides = {"num_hidden_layers": args.layers}
    result: dict = {"layers": args.layers, "prompts": [p["prompt_id"] for p in prompts], "lengths": [p["length"] for p in prompts],
                    "decode_steps": args.decode_steps, "timings_s": {}, "memory": {}}

    # 1. from_pretrained, resident (host memory, then the device): transformers' loader on the whole truncated model.
    started = time.perf_counter()
    config = transformers.AutoConfig.from_pretrained(model_config["repository"], revision=model_config["revision"])
    for key, value in overrides.items():
        setattr(config, key, value)
    resident = transformers.AutoModelForCausalLM.from_pretrained(
        model_config["repository"], revision=model_config["revision"], config=config, dtype=dtype,
        experts_implementation=profile.experts_implementation, attn_implementation=profile.attention_implementation,
    )
    resident = resident.to(device).eval()
    result["timings_s"]["from_pretrained"] = time.perf_counter() - started
    profile.check_model(resident)
    expected_weights = weights_digest(resident)
    recorder = StepRecorder(resident, adapter)
    handles = observe_resident_experts(resident, recorder)
    started = time.perf_counter()
    expected = run_prompts(resident, recorder, prompts, args.decode_steps, device)
    result["timings_s"]["resident_run"] = time.perf_counter() - started
    result["memory"]["resident_peak_device_bytes"] = torch.cuda.max_memory_allocated(device)
    for handle in handles:
        handle.remove()
    del resident, recorder, handles
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 2. The streaming reference on the same truncation.
    loaded: dict[str, list] = {}

    def digest_loaded(name, module):  # once per layer: later loads of it are the same bytes
        for parameter, tensor in module.named_parameters(recurse=False):
            if f"{name}.{parameter}" not in loaded:
                loaded[f"{name}.{parameter}"] = [str(tensor.dtype), list(tensor.shape), tensor_sha256(tensor)]

    started = time.perf_counter()
    reference = StreamingReference(
        model_config["repository"], model_config["revision"], dtype, device, experts_implementation=profile.experts_implementation,
        attn_implementation=profile.attention_implementation, config_overrides=overrides, on_load=digest_loaded,
    ).load()
    result["timings_s"]["streaming_load"] = time.perf_counter() - started
    profile.check_model(reference.model)
    recorder = StepRecorder(reference.model, adapter)
    reference.on_call = recorder.on_call
    started = time.perf_counter()
    streamed = run_prompts(reference.model, recorder, prompts, args.decode_steps, device)
    result["timings_s"]["streaming_run"] = time.perf_counter() - started
    result["memory"]["streaming_peak_device_bytes"] = torch.cuda.max_memory_allocated(device)
    result["memory"]["streaming_peak_expert_device_bytes"] = reference.peak_expert_device_bytes
    result["memory"]["process"] = process_memory()
    got_weights = {**weights_digest(reference.model), **loaded}

    # Verdicts.
    weight_names = sorted(set(expected_weights) | set(got_weights))
    differing_weights = [name for name in weight_names if expected_weights.get(name) != got_weights.get(name)]
    steps = []
    for a, b in zip(streamed, expected, strict=True):
        verdict = compare(a, b)
        steps.append({"prompt_id": a["prompt_id"], "step": a["step"], **verdict})
    result.update(
        weights_compared=len(weight_names),
        expert_tensors_compared=len(loaded),
        differing_weights=differing_weights,
        steps_compared=len(steps),
        steps_equal=sum(all(s["matches"].values()) for s in steps),
        expert_outputs_checked=sum(s["expert_outputs_checked"] for s in steps),
        quantities=sorted(steps[0]["matches"]) if steps else [],
        passed=not differing_weights and all(all(s["matches"].values()) for s in steps) and bool(steps),
        experts_implementation=reference.model.config._experts_implementation,
        attention_implementation=reference.model.config._attn_implementation,
        prompts_provenance=provenance,
    )
    with JsonlWriter(output / "resident.jsonl.gz") as log:
        for record in expected:
            log.write(record)
    with JsonlWriter(output / "streaming.jsonl.gz") as log:
        for record in streamed:
            log.write(record)
    (output / "check.json").write_text(json.dumps({**result, "steps": steps}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("prompts_provenance",)}, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
