# Initial reading notes (2026-08-22)

## Evidence boundary

- The primary paper is arXiv:2608.03893, submitted 2026-08-04.
- No official implementation is linked by the paper. `external/kvbridge` is an Apache-2.0 independent reproduction; its README explicitly says its checked-in Qwen plans match the paper dimensions but its author did not validate the 14B/32B headline results.
- KVBridge, RULER, lm-evaluation-harness, LatentMAS, and C2C are pinned in `notes/SOURCES.md`.

## What the paper actually establishes

- Calibration: 500 FineWeb-Edu sequences x 1,024 tokens, stride 4 (~128K observations).
- Qwen3 14B->32B uses top-k=8 and reports a ~1.07B-parameter / ~4 GB FP32 mapper.
- Multi-turn: only Qwen3 14B<->32B, 100 CoQA conversations (~15 turns), plotted/scored through ten turns.
- The paper reports small-to-large gap widening by 1.7 pp from turns 1 to 10 and large-to-small drift increasing by 0.33 pp/turn. It explicitly says the latter could accumulate in very long sessions.
- It does not report pure repeated cache cycles without token generation, 16/32 cycles, cycle singular values, per-token exposure, or a native-prefix identity path.

## Code fit for the proposed project

- `CrossModelKVMapper.map` is token-separable after source RoPE removal: features flatten batch x tokens and project each token independently.
- The mapper accepts any cache token length; suffix-only work mainly needs cache/range slicing plus target rotary factors computed at the suffix's absolute positions.
- The current Hugging Face helper performs a single full-prefix handoff. It has no repeated A<->B orchestrator, native-cache ledger, or suffix merge path.
- Therefore the proposed additions are genuinely new orchestration/state-management code, while ridge fitting, layer selection, cache conversion, RoPE helpers, and diagnostics can be reused.

## Important methodological caveat

The native suffix produced by model B is conditional on the cache B received. After mapping that suffix back to A, the resulting A cache is a hybrid: exact old A prefix plus mapped B-produced suffix. This is valid as an inference intervention, but it is not identical to full-prefill A. Teacher-forced oracle comparison is essential to quantify this boundary error separately from recursive remapping.

## Recommended first implementation gates

1. Reproduce KVBridge's pinned Qwen3 0.6B->1.7B T2 smoke job on this GPU.
2. Fit both directions; add a pure cache-cycle runner over h={1,2,4,8,16,32}.
3. Add token slicing and absolute-position RoPE tests before implementing the ledger.
4. Run teacher-forced alternating chunks with identical tokens for oracle, full-remap, and native-cache skip.

## Local T2 pilot result

- Hardware: NVIDIA RTX 3090 24 GB; PyTorch 2.13.0+cu130.
- Calibration capture: 16/16 sequences x 512 tokens; all 16 shard sizes and SHA-256 hashes validated.
- Fit: 81.74 s; BF16 artifact; mean fit R² K=0.88824, V=0.83432.
- Held-out (8 sequences): cache R²=0.23369, attention cosine mean=0.65678, logit KL mean=1.94694, next-token agreement=0.25.
- Both repository gates fail (attention >=0.90 and KL <=0.20). This pair remains a debugging-only pilot.
- PyTorch 2.13/Triton triggers a `PyGILState_Release` fatal error during interpreter finalization after capture/evaluation outputs have been fully written. Artifact and manifest integrity were checked independently; pinning a known-stable PyTorch version is advisable before production runs.

## Local Qwen3 1.7B -> 4B result

- Used KVBridge's pinned T2 headline config: 24 x 512-token calibration sequences, stride 4, top-k=2; 16 held-out 512-token sequences.
- Calibration capture: 24/24 shards, 3.1 GB total; sizes and SHA-256 hashes validated.
- Fit: 173.91 s; BF16 artifact; mean fit R² K=0.92151, V=0.89196.
- Held-out: cache R²=0.56913, attention cosine mean=0.52990, logit KL mean=1.22532, next-token agreement=0.75.
- Transfer median=16.91 ms versus target prefill median=84.51 ms (5.04x ratio).
- Both quality gates fail (attention >=0.90, KL <=0.20). Despite higher fit R² and next-token agreement than 0.6B->1.7B, attention-sensitive fidelity remains inadequate for a clean repeated-composition study.

## Ministral 3B -> 8B sequential code audit

- The paper checkpoints are the Reasoning 2512 models. Standard HF shards are 7.70 GB and 17.84 GB; each repository also carries a duplicate consolidated checkpoint, which is excluded.
- KVBridge required support for nested multimodal `text_config`, nested decoder layers/rotary modules, Transformers 5, Ministral query position scaling, and corrected Mistral tokenizer regex.
- Added tests cover multimodal decoder discovery and post-RoPE query scaling. Full suite: 54 passed, 1 CUDA skip. A tiny real `Ministral3ForCausalLM` forward verifies cache/query shapes and YaRN strip/reapply error below 3e-8.
- Sequential capture generated aligned, revision-pinned, hash-validated source/target shards without model co-residency.
- The initial 16-sequence k=4 pilot was invalid: 2,048 stride-sampled observations for a 4,096-dimensional regression. It interpolated calibration (reported fit R² ~1) and collapsed held-out (cache R² -509.32, attention cosine 0.3521, KL 4.4085, top-1 agreement 12.5%). These are underdetermination diagnostics, not a model-pair conclusion.
- Added a fail-fast gate requiring observations to exceed mapper feature dimension. A valid k=4 pilot needs at least 33 x 512-token sequences at stride 4; 64 sequences is the recommended next run. The paper's k=all feature dimension is 26 x 8 x 128 = 26,624, explaining its much larger ~128K-observation calibration.

## Valid Ministral 3B -> 8B k=4 pilot

- Calibration: 64 x 512 tokens, stride 4 = 8,192 observations for 4,096 features. All 64 sequentially captured pair shards passed the fail-closed manifest/hash checks.
- Fit: 390.80 s on RTX 3090; 285.3M-parameter mapper stored in BF16; K R²=0.92458, V R²=0.85613.
- Fixed held-out set (8 x 512 tokens): cache R²=0.76380, attention cosine mean=0.67620, logit KL mean=0.30877, next-token agreement=87.5%.
- This is a large recovery over the invalid underdetermined run (attention 0.352, KL 4.409), validating the sample-size guard and sequential pipeline.
- It still fails KVBridge's strict single-hop gates (attention >=0.90 and mean KL <=0.20). It is a k=4 pilot, not the paper's k=all / ~128K-observation setting, so it cannot reproduce or refute the paper's reported Ministral retention.
