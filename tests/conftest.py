from __future__ import annotations

from pathlib import Path

import pytest
import torch

from awpmi.runtime import configure_reproducible_numerics

# Before any CUDA matmul: the certificate's accumulation model assumes these flags.
configure_reproducible_numerics()

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "smollm2-135m.yaml"

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

PROMPTS = (
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return",
    "Once upon a time, there was a",
    " The game 's battle system , the BliTZ system , is carried over directly from Valkyira Chronicles",
    "1, 2, 3, 4, 5, 6, 7,",
    "Water boils at a temperature of 100 degrees",
)


@pytest.fixture(scope="session")
def experiment_config():
    from awpmi.config import load_config

    return load_config(CONFIG_PATH)[0]


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def loaded_model(experiment_config, device):
    from awpmi.models.smollm2 import ModelSpec, load_model, resolve_dtype

    spec = ModelSpec(
        repository=experiment_config.model.repository,
        revision=experiment_config.model.revision,
        dtype=resolve_dtype(experiment_config.model.dtype, device),
        device=device,
    )
    return load_model(spec)


@pytest.fixture(scope="session")
def model(loaded_model):
    return loaded_model[0]


@pytest.fixture(scope="session")
def tokenizer(loaded_model):
    return loaded_model[1]


@pytest.fixture(scope="session")
def prompt_ids(tokenizer, device):
    return [tokenizer(text, return_tensors="pt").input_ids.to(device) for text in PROMPTS]


@pytest.fixture(scope="session")
def reference_numerics(model, experiment_config):
    from awpmi.bounds.residual import ReferenceNumerics
    from awpmi.models.smollm2 import lm_head_weight

    weight = lm_head_weight(model)
    return ReferenceNumerics(
        output_dtype=weight.dtype,
        accumulation_unit_roundoff=experiment_config.numerics.accumulation_unit_roundoff,
        reduction_length=weight.shape[1],
    )
