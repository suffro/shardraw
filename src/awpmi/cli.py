"""Command line (roadmap §3.2): `awpmi pack`.

    uv run awpmi pack lm-head [--config configs/phase3-storage.yaml] [--output DIR]
    uv run awpmi pack experts [--config configs/phase3-moe.yaml] [--output DIR]

lm-head  the refinement pack of the configured decomposition of a model's LM head: level records
         and remainder norms in an AWPMI safetensors file, exact rows referring to the published
         checkpoint tensor (checked byte for byte against the model's LM-head weight).
experts  the expert pack of a mixture-of-experts model: every expert-sliced parameter, referring
         to the published checkpoint wherever it holds the same bytes.

Either writes `manifest.json` (files with sizes and sha256, segments with offsets and sha256,
source model and revision, packing configuration) and prints a summary. The benchmarks re-open
packs with every segment re-hashed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def pack_lm_head(config_path: Path, output: Path | None) -> dict:
    import torch

    from awpmi.config import load_config
    from awpmi.decomposition import RefinementDecomposition
    from awpmi.models.smollm2 import ModelSpec, lm_head_weight, load_model, resolve_dtype
    from awpmi.storage.pack import SourceFile
    from awpmi.stores.refinement import write_refinement_pack

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config, _ = load_config(REPO_ROOT / config["source_run"] / "config.yaml")
    device = torch.device(model_config.model.device if torch.cuda.is_available() else "cpu")
    spec = ModelSpec(model_config.model.repository, model_config.model.revision, resolve_dtype(model_config.model.dtype, device), device)
    model, _ = load_model(spec)
    decomposition = RefinementDecomposition.build(lm_head_weight(model), config["decomposition"])
    source = SourceFile(spec.repository, spec.revision, config["pack"]["source_file"])
    pack = write_refinement_pack(
        decomposition,
        output or REPO_ROOT / config["pack"]["directory"],
        source=(source, config["pack"]["source_tensor"]),
        packing={"tool": "awpmi pack lm-head", "config": str(config_path)},
    )
    return pack.manifest


def pack_experts(config_path: Path, output: Path | None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from awpmi.models.moe import write_expert_pack
    from awpmi.storage.pack import SourceFile

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(config["repository"], revision=config["revision"], dtype=dtype)
    source = SourceFile(config["repository"], config["revision"], config["source_file"])
    directory = output or REPO_ROOT / yaml.safe_load(config_path.read_text(encoding="utf-8"))["pack"]["directory"]
    pack = write_expert_pack(model, directory, sources={"checkpoint": (source, None)}, packing={"tool": "awpmi pack experts"})
    return pack.manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="awpmi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="write a pack and its manifest")
    pack.add_argument("kind", choices=["lm-head", "experts"])
    pack.add_argument("--config", default=None)
    pack.add_argument("--output", default=None, help="pack directory (default: the config's)")
    args = parser.parse_args(argv)
    default = {"lm-head": "phase3-storage.yaml", "experts": "phase3-moe.yaml"}[args.kind]
    config = Path(args.config) if args.config else REPO_ROOT / "configs" / default
    output = Path(args.output) if args.output else None
    manifest = (pack_lm_head if args.kind == "lm-head" else pack_experts)(config, output)
    summary = {
        "kind": manifest["kind"],
        "files": {key: {k: v for k, v in entry.items() if k != "sha256"} for key, entry in manifest["files"].items()},
        "segments": len(manifest["segments"]),
        "bytes": sum(entry["rows"] * entry["row_bytes"] for entry in manifest["segments"].values()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
