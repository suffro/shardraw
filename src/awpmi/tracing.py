"""Raw structured traces and the environment metadata every benchmark must save."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch

TRACKED_PACKAGES = ("torch", "transformers", "safetensors", "numpy", "huggingface-hub", "tokenizers", "pyarrow")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_TREES = ("src", "benchmarks", "configs")


def source_tree_sha256(repo_root: Path) -> str:
    """Hash of the code and configs that produced a run, independent of git state."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for tree in SOURCE_TREES
        for path in (repo_root / tree).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in files:
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(repo_root: Path, model: Mapping[str, Any], numerics_flags: Mapping[str, Any]) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    lock = repo_root / "uv.lock"
    status = _git(repo_root, "status", "--porcelain")
    return {
        "model": dict(model),
        "python": sys.version,
        "os": platform.platform(),
        "packages": {name: importlib_metadata.version(name) for name in TRACKED_PACKAGES},
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "numerics": dict(numerics_flags),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "source_tree_sha256": source_tree_sha256(repo_root),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
    }


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self._handle = open(path, "w", encoding="utf-8", newline="\n")

    def write(self, record: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_digest(records: Iterable[Mapping[str, Any]], exclude_keys: Iterable[str] = ()) -> str:
    """sha256 over records with non-deterministic fields (e.g. timings) removed."""
    excluded = set(exclude_keys)
    digest = hashlib.sha256()
    for record in records:
        kept = {key: value for key, value in record.items() if key not in excluded}
        digest.update(json.dumps(kept, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
