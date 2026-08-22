# Cross-Model KV Compositional Drift

Reproducible workspace for testing whether locally accurate cross-model KV
transfer remains stable under repeated composition.

The current 24 GB main pair is **Qwen2.5-1.5B Base ↔ Qwen2.5-7B Base**.
Both have 28 layers and 128-dimensional attention heads; their KV-head counts
differ (2 vs 4), making the large→small→large rank bottleneck directly testable.

## Current status

- Shared-tokenizer gate: passed on 10,000 FineWeb-Edu texts / 6,175,046 tokens.
- Forward calibration: 500 × 1024 tokens, stride 4 (128,000 observations).
- Qwen 1.5B→7B k=28 fit: complete; mean fit R² K=0.8111, V=0.6582.
- Next gate: held-out single-hop fidelity, then HellaSwag/ARC retention.
- Repeated-cycle tests run only if single-hop retention passes the preregistered gate.

## Layout

- `external/kvbridge`: pinned research fork/branch
- `configs/`: revision-bound experiment contracts
- `scripts/`: sequential capture, evidence indexing, and tokenizer validation
- `notes/`: provenance and reading notes

Models, calibration shards, mapper weights, and paper copies are intentionally
excluded from Git. Small JSON run records are retained.

Large-model experiments use sequential model loading on one 24 GB RTX 3090.
