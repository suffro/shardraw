"""Declared numerical environment shared by the reference and AWPMI.

The certificate reasons about how the reference computes its logits (see
`awpmi.bounds.residual.ReferenceNumerics`). That reasoning is only valid if the
backend actually accumulates in FP32, so the environment is fixed here and
recorded with every run instead of being left to library defaults.
"""

from __future__ import annotations

import os

import torch

CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_reproducible_numerics() -> dict[str, object]:
    """Fix determinism and accumulation-precision flags; return them for tracing.

    Must run before the first CUDA matmul so that `CUBLAS_WORKSPACE_CONFIG` takes effect.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Default True: cuBLAS may then reduce split-K partial sums in BF16/FP16, for
    # which no useful accumulation-error bound exists. Split-K itself stays allowed:
    # with FP32 reductions it is just another summation order, covered by γ_n.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    return describe_numerics()


def describe_numerics() -> dict[str, object]:
    matmul = torch.backends.cuda.matmul
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k": matmul.allow_bf16_reduced_precision_reduction_split_k,
        "cuda_matmul_allow_fp16_accumulation": matmul.allow_fp16_accumulation,
    }
