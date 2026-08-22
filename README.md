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
- Held-out 32-prefix state smoke: cache R² 0.9913, probe attention cosine
  0.6961, logit KL 0.1866, top-1 agreement 81.25%.
- HellaSwag 1K: native/transfer acc 52.4/47.9 (91.4% retention) and
  acc_norm 67.7/60.6 (89.5% retention). This is a borderline stress pair, not
  yet a near-lossless main pair.
- Next gate: paired confidence intervals and ARC-Challenge; repeated cycles are
  still blocked by the single-hop gate.
- Repeated-cycle tests run only if single-hop retention passes the preregistered gate.

## Layout

- `external/kvbridge`: pinned research fork/branch
- `configs/`: revision-bound experiment contracts
- `scripts/`: sequential capture, evidence indexing, and tokenizer validation
- `notes/`: provenance and reading notes

Models, calibration shards, mapper weights, and paper copies are intentionally
excluded from Git. Small JSON run records are retained.

Large-model experiments use sequential model loading on one 24 GB RTX 3090.
