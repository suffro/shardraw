# Conventions

## Repository conventions

- Python ≥ 3.11 (the environment uses 3.13, see `.python-version`), managed with `uv`.
  Dependencies are pinned in `pyproject.toml` and `uv.lock`. Torch comes from the
  PyTorch cu130 index (`[tool.uv.sources]`). If `uv` is not on PATH, use `python -m uv`.
- The model revision and the dataset revision are pinned by commit hash in
  `configs/smollm2-135m.yaml`.
- Every benchmark writes to its own `experiments/phase<N>/<run-name>/`: config,
  environment metadata (versions, GPU, numerics flags, git state, source-tree hash,
  dataset sha256), prompts, raw per-run records, validation records, and a digest. Raw
  results are kept in the repository.

## Development workflow

- Tests: `python -m uv run pytest`. Tests marked `model` load the pinned SmolLM2
  (downloaded on first use).
- Benchmark: `python -m uv run python benchmarks/run.py --output experiments/phase1/<name>`.
  Add `--num-prompts N` for a quick run.
- Report and gate: `python -m uv run python benchmarks/report.py <run> --compare <second run>`.
  Reproducibility is judged by comparing the digests of two identical runs.
- Ordering vs. bound diagnosis:
  `python -m uv run python benchmarks/oracle.py <run> --output <run>-oracle`.

## Important rules

- No heuristic or estimate may participate in `certified=True`. Bounds are upper bounds
  on true values, including floating-point error, and every new bound needs a
  soundness test (exact `Fraction` arithmetic where feasible).
- When a guard is added, confirm once that it fails when disabled (see decision 0001).
- Never change the reference model to make AWPMI agree with it. Numerical-environment
  flags apply to both sides and are recorded.
- Benchmarks stop at the first hard failure (certified or fallback mismatch, envelope
  violation, non-bitwise fallback, prefix mismatch) and save it to `failure.json`.
