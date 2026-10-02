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
- Phase 1B decomposition oracle:
  `python -m uv run python benchmarks/refinement_oracle.py --output experiments/phase1b/<name>`.
  It reuses the prompts of `source_run` in `configs/phase1b-refinement.yaml`. Then
  `python -m uv run python benchmarks/refinement_report.py <run> --compare <second run>`.
- Phase 1C runtime benchmark:
  `python -m uv run python benchmarks/refinement_runtime.py --output experiments/phase1c/<name>`
  (config `configs/phase1c-runtime.yaml`, prompts of its `source_run`, comparison with its
  `oracle_run`). Then
  `python -m uv run python benchmarks/refinement_runtime_report.py <run> --compare <second run>`.
  Fallback evidence: `python -m uv run python benchmarks/fallback_study.py --output <dir>`.
- Timing fields (`timings_ms`, `reference_ms`, `profile.json`) are recorded but excluded
  from digests. Runs that are compared for reproducibility must use the same source tree.
- Large raw record files are written as reproducible gzip (`*.jsonl.gz`, mtime 0);
  `read_jsonl` reads both forms. Digests are computed over the decoded records.

## Important rules

- No heuristic or estimate may participate in `certified=True`. Bounds are upper bounds
  on true values, including floating-point error, and every new bound needs a
  soundness test (exact `Fraction` arithmetic where feasible).
- When a guard is added, confirm once that it fails when disabled (see decision 0001).
- Never change the reference model to make AWPMI agree with it. Numerical-environment
  flags apply to both sides and are recorded.
- Benchmarks stop at the first hard failure (certified or fallback mismatch, envelope
  violation, non-bitwise fallback, prefix mismatch) and save it to `failure.json`.
- Byte savings are reported as *effective* bytes: every resident metadata byte, every
  scale, and the fallback's reads count (decision 0003). Never report page counts
  alone. A runtime's bytes are what its store's read log says, and must equal the
  decomposition's accounting for the same rows (decision 0004).
- A runtime reads weight values only through a store. Bounds for the runtime's own
  arithmetic live in `awpmi.bounds` and are validated twice: against exact rational
  arithmetic in tests, and against float64 on every benchmark prompt.
- Decision-gate thresholds are fixed in config before a full run, not after seeing it.
