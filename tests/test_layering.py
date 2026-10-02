"""Phase 3 layering (decision 0006): the storage core knows no model and no certificate; certification knows no storage."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "awpmi"
CORE = ("storage", "streaming", "materialization")
# Names that would tie the core to one model, one tensor layout or one routing implementation.
MODEL_WORDS = re.compile(
    r"smollm|granite|llama|mixtral|qwen|deepseek|olmoe|gpt.?oss|lm_head|embed_tokens|gate_up_proj|down_proj|"
    r"block_sparse_moe|router|top_k_index",
    re.IGNORECASE,
)
CORE_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+(transformers|awpmi\.models|awpmi\.bounds|awpmi\.certificate|awpmi\.refinement_head|"
    r"awpmi\.suffix_runtime|awpmi\.stores|awpmi\.oracle|awpmi\.decomposition)\b",
    re.MULTILINE,
)
CERTIFICATION_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+awpmi\.(storage|streaming|materialization)\b", re.MULTILINE
)


def core_files():
    for package in CORE:
        yield from sorted((SRC / package).glob("*.py"))


def test_the_storage_core_knows_no_model():
    files = list(core_files())
    assert len(files) >= 8
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not MODEL_WORDS.search(text), f"{path.name}: {MODEL_WORDS.search(text).group(0)!r}"
        assert not CORE_FORBIDDEN_IMPORTS.search(text), f"{path.name} imports a layer above it"


def test_certification_does_not_depend_on_storage():
    for path in [*sorted((SRC / "bounds").glob("*.py")), SRC / "certificate.py", SRC / "refinement_head.py"]:
        assert not CERTIFICATION_FORBIDDEN_IMPORTS.search(path.read_text(encoding="utf-8")), path.name
