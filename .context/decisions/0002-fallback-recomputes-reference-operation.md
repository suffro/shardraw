# 0002 — Fallback recomputes the reference operation from the materialized pages

Status: accepted (Phase 1, 2026-10-01)

## Context

When the certificate stays UNKNOWN after every page has been materialized, AWPMI
still needs a token. The float64 page-accumulated logits are numerically close to
the reference but are a different computation. When two reference logits round to
the same BF16 value, their argmax can legitimately differ from the reference's.

## Decision

The fallback reassembles the weight from the materialized pages in column order
and applies `torch.nn.functional.linear(hidden, weight)`. This is the operation
`nn.Linear(bias=False)` performs in the reference, with the same input tensor
shape and dtype. On this setup it reproduces the reference logits **bitwise**. Every
benchmark checks this on every prompt and page width (`fallback_bitwise_equal`),
and a mismatch is a hard failure.

The float64 page-accumulated logits are used only for certification and for the
"AWPMI logits ≈ reference" check. That check uses the derived envelope as its
tolerance.

**Extended in decision 0004.** The Phase 1C runtime keeps this fallback as its default
(`FallbackMode.FULL`). It also offers an opt-in `FallbackMode.MASKED`, which runs the
same operation with the full shape on the surviving rows only. That mode relies on the
GEMM's row independence, and it is self-tested, guarded and evidenced.

## Rejected

- *Taking argmax of the float64 page-accumulated logits as the fallback.* It can
  disagree with the reference on BF16 ties, which would produce fallback
  mismatches.
- *Calling `model.lm_head` directly.* It would bypass the materialized pages, so it
  would not exercise the path that Phase 3 replaces with real storage.
