# Local checkpoint deployment

Run `ba52f0c8dafd4a1880b60d5c557ec35e` completed with nine Luminal calls,
eight real model experiments, and one failed response. This export and final
comparison used **zero additional API calls**. `runs/experiments.jsonl` is
unchanged (SHA-256 `4531fde279ca018d73f769bfbbd815169b5618bc3bdac77b9317d01f8418eb0a`).

## Reproduced validation winner

Round 2, classifier role: `StandardScaler` followed by `LogisticRegression`,
`C=0.1`, `class_weight=None`, `solver=lbfgs`, `max_iter=1000`, seed 42.
The saved source-group manifest is reused; no partitions are regenerated.
Only 1,148 training clips fit the scaler/classifier. Each clip has equal total
training weight distributed over its windows by actual duration.

Audio is decoded locally, downmixed to mono, resampled to 16 kHz, divided into
non-overlapping four-second windows, and zero padded at the end. All 52 features
are used: means/stds of 20 MFCCs and centroid, bandwidth, rolloff, flatness, ZCR,
and RMS. Spectrogram policy: FFT 512, hop 160, 64 Mel bands, power 2,
centered constant padding, 0–8 kHz; dB reference 1, floor 1e-10, top 80 dB.
Clip scores are duration-weighted segment outputs; real=0 and fake=1.
They are uncalibrated scores. Inference's existing ±0.05 abstention band remains.

Validation balanced accuracy reproduced **exactly**, 0.9568335941575378
(baseline 0.9510954616588418), on 250 clips. Validation synthetic F1 is
0.9497716894977168, confusion matrix `[[135,7],[4,104]]`, ROC AUC
0.9897626499739176. Threshold is frozen at **0.39278538241090233**.
All discrete metrics must match exactly; threshold and ROC AUC allow absolute
roundoff of at most 1e-9. No mismatch occurred. Export also verifies the
pipeline's inference feature layout against the original metrics.

Checkpoint: `models/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e.joblib`.
It includes the fitted scaler/classifier, label map, full feature settings,
decision threshold, seed, training/validation groups, split hash, dataset revision,
configuration, source event, validation reproduction, and library versions.
Readable metadata is in the corresponding `runs/optimized-v1-.../checkpoint_metadata.json`.
Baseline `models/baseline.joblib` was preserved byte for byte.
Only load trusted, locally generated joblib files; they contain pickle data.

## Final test comparison and history

Before comparison there were no saved test evaluation reports or started markers,
and none of the 252 retained test clips had feature caches. The prior saved
inference checks used validation examples. Preparation did decode all data for
integrity and grouping; this is not model evaluation. These observations support
no prior recorded model test evaluation, but cannot exclude unlogged/deleted work.
The pre-comparison audit is preserved as `test_history_before_comparison.json`.

Both models were frozen before test evaluation and used their validation
thresholds. Each model was evaluated once; further evaluation commands replay
saved metrics. No test-based hyperparameter or threshold tuning was performed.
Metrics describe threshold classifications, before the UI abstention rule.

| Model | Test balanced accuracy | Synthetic F1 | Confusion matrix |
| --- | ---: | ---: | --- |
| Baseline | 96.63501636% | 0.9596412556 | `[[136,7],[2,107]]` |
| Validation winner | 96.41688587% | 0.9589041096 | `[[138,5],[4,105]]` |

Rows are actual real/fake; columns are predicted real/fake. Both have nine
errors and 96.42857143% ordinary accuracy. The winner's balanced accuracy is
0.21813049 percentage points lower, and F1 is lower by 0.0007371460.
There is no evidence here of a test improvement; keep the baseline as default.
This small English dataset has source/domain bias, and no distinct TTS vendor
was held out of optimization. These results do not establish unseen-generator
generalization or forensic authenticity.

## Opt in or roll back

From the repository root, explicitly select the exported checkpoint for a demo:

```sh
ECHOPROOF_MODEL_PATH="$PWD/models/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e.joblib" .venv/bin/streamlit run app.py
```

Stop the running app before changing its environment. Roll back with:

```sh
ECHOPROOF_MODEL_PATH="$PWD/models/baseline.joblib" .venv/bin/streamlit run app.py
```

With `ECHOPROOF_MODEL_PATH` unset, the backend defaults to the baseline. Missing
or incompatible explicit checkpoints return `unavailable` rather than substitute
a model silently. The stable seven-field inference contract is unchanged;
`model_version` correctly identifies `optimized-v1` when it is selected.
No frontend files or shared dependencies were changed.

Offline replay/export and saved test comparison:

```sh
.venv/bin/python -m scripts.promote_winner --run-id ba52f0c8dafd4a1880b60d5c557ec35e
.venv/bin/python -m scripts.compare_models --winner models/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e.joblib --winner-run-dir runs/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e
```

The exporter refuses baseline replacement, mismatched provenance, unsupported
feature/classifier families, and validation mismatch. Existing exports replay
without refitting. It does not access test or construct a provider client.
The final evaluator's started markers prevent automatic repeated test access
after a partial failure. Preserve the original reports when investigating one.

## Failed response investigation

The eighth call, round 3 classifier, failed at `2026-10-11T20:18:25.745002+00:00`
with `Luminal returned empty or oversized chat content; no automatic retry was made.`
No proposal was accepted or fit for that call. Existing logs did not save byte
counts, completion reason, or token usage, so they cannot identify whether the
response was empty, missing/non-text, or oversized, or establish a provider cause.
There is no evidence to attribute it to quota, authentication, or token exhaustion.

Future errors distinguish missing text, empty text, oversized text, and absent
choices. Safe error reasons include bounded counts, a whitelisted completion
reason, content kind/size, refusal/tool-call presence, and usage/reasoning token
counts when available. Response text, reasoning text, IDs, headers, credentials,
and arbitrary provider error text are excluded. These diagnostics flow through
the existing event `reason` field, preserving the dashboard event schema.
SDK retries remain disabled; no fallback or diagnostic API request is made.
Tests use mocked HTTP transports only.

## Verification and integration note

The targeted promotion/model/agent suite passed all 88 tests. The ordinary full
suite had 147 passes and one frontend smoke failure: it assumes no local model
but a baseline exists, so silent audio returns `invalid` instead of `unavailable`.
The original committed backend was checked and returned the same `invalid`
result. No frontend files were changed. With `ECHOPROOF_MODEL_PATH` explicitly
pointing to a nonexistent checkpoint, all 148 tests passed, including a real
local winner checkpoint load/inference check on validation speech. There is one
existing Librosa deprecation warning. The frontend owner can isolate its
missing-backend smoke test from local model files at the next integration step.
Offline export and comparison replay were verified without fitting/rescoring;
baseline, split manifest, and live experiment history hashes are unchanged.
