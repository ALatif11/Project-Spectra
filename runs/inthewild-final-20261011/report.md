# Final InTheWild hackathon demonstration — 2026-10-11

The selected candidate was frozen before the sole authorized evaluation of the local sealed 1,800-recording test partition. All 1,800 recordings were scored once: 900 human, 900 synthetic, across 11 speakers disjoint from the 35 training and 8 validation speakers. No production model fitting, test-driven threshold/hyperparameter changes, Luminal calls, or MongoDB integration occurred. Optimization is stopped.

## Actual final results

| Metric | Result |
| --- | ---: |
| Balanced accuracy | 93.50% |
| Synthetic precision | 91.87% |
| Synthetic recall | 95.44% |
| Synthetic F1 | 93.62% |
| False-positive rate | 8.44% |
| False-negative rate | 4.56% |

Confusion matrix, rows = actual class, columns = predicted class:

| | Predicted human | Predicted synthetic |
| --- | ---: | ---: |
| Actual human | 824 | 76 |
| Actual synthetic | 41 | 859 |

Synthetic is the positive class. Binary decisions use the unchanged validation-selected threshold **0.41431927302943794**. The app's fixed ±0.05 abstention band produces an inconclusive label near that threshold; the binary benchmark metrics above do not apply that band. Scores remain uncalibrated model outputs, not probabilities of authenticity.

Machine-readable results: `/Users/davidbonakdar/Documents/GitHub/Project-Spectra/runs/inthewild-final-20261011/test_metrics.json`. Canonical partition result: `/Users/davidbonakdar/Documents/GitHub/Project-Spectra/data/inthewild/experiment-12000-v1/sealed/test_metrics.json`. Per-recording segment scores and labels are in `/Users/davidbonakdar/Documents/GitHub/Project-Spectra/data/inthewild/experiment-12000-v1/sealed/test_predictions.jsonl`; per-speaker metrics are included in the result JSON. Only saved results should be replayed: do not retrain, tune or rescore this exposed test set.

## Frozen checkpoint and preprocessing

Selected and frozen checkpoint SHA-256: `e7263a3a4a1c789ffcd3765f1fb12ec2d41f0334057f57950c75a1c3d0a448bc`. The frozen copy is `/Users/davidbonakdar/Documents/GitHub/Project-Spectra/models/inthewild-final-frozen.joblib`; the original `models/inthewild-v1-12000.joblib` is byte-identical and preserved. The frozen StandardScaler → logistic regression pipeline uses all 52 original features, C=1, no class weights, lbfgs, max_iter=1000, seed 42, and the saved fitted scaler. The validation balanced accuracy remains **92.92057374443738%**.

Preprocessing is shared with training: local float32 decode, arithmetic mono downmix, resampling to 16 kHz, no loudness normalization, non-overlapping four-second windows, final zero padding, 20 MFCC means and standard deviations plus six spectral/energy mean/std pairs. Log-Mel uses FFT 512, hop 160, 64 bands. Recording scores are duration-weighted means of segment outputs. Label order is verified: bonafide=0, spoof=1. Frozen model/configuration/source hashes are recorded in `/Users/davidbonakdar/Documents/GitHub/Project-Spectra/runs/inthewild-final-20261011/freeze.json` and checked before and after test scoring.

## Seal and grouping audit

Existing verification and research records report no prior test scoring. Before this evaluation, the feature cache contained exactly 10,200 optimization entries and no test entries. The authorized preflight confirmed all test speakers were disjoint, both classes present, and no duplicate path, utterance ID, encoded-audio hash or decoded-PCM hash across the combined 12,000 selected recordings. Test membership matched the original preparation counts and per-speaker distribution. Every selected test clip's upstream identity, label, encoded bytes and decoded PCM were checked against its saved preparation record during evaluation.

No historical digest of sealed/test.json was recorded. Existing logs and optimization-only caches support no prior test scoring; they are not an OS access audit. Preparation decoded all clips for PCM identity checks; shard extraction also read batches containing unselected audio, without extracting/scoring test features.

Test manifest SHA-256 established at final preflight and preserved through evaluation: `38c2075ab91444839d294caf7f9ca963672821af3eb496af0273c7f68db6a368`. An exclusive partition-level `test_started.json` blocks repeat attempts even with a new output directory; successful repeats only return saved metrics without reopening or scoring test audio. Partial failures cannot retry automatically. The marker and prediction journal are retained.

Speaker-disjoint local partitions; source recording/session, TTS vendor, shared text and near-duplicate provenance are unavailable. Cross-speaker source leakage cannot be ruled out. Public-figure English web audio is not representative of arbitrary microphones/languages. Uncalibrated scores are not forensic proof. Upstream test is repurposed for this experiment and is no longer an independent benchmark.

Whole-speaker grouping does not establish recording/session, shared-text or unseen-generator independence. Exact duplicates were excluded during preparation, including speakers connected by cross-speaker duplicates, but near duplicates and shared source audio across different speakers cannot be ruled out. The test contains only 11 public-figure speakers with uneven counts; per-speaker balanced accuracy ranges from 73.93% to 100%. These local held-out results are not the untouched published InTheWild benchmark, and do not establish reliability on arbitrary microphones, languages, codecs or generators. No independent TTS-vendor holdout was performed. Research estimates are not forensic proof.

## Launch and reversible rollback

From the repository root, launch the final demonstration:

```sh
.venv/bin/python -m scripts.launch_final_demo
```

The launcher verifies the frozen checkpoint's digest and compatibility, selects it through the existing `ECHOPROOF_MODEL_PATH` interface and starts Streamlit. It does not overwrite any model or change the default baseline. No frontend/interface/dependency changes were required.

Stop the demo with Ctrl-C, then use the preserved baseline copy:

```sh
.venv/bin/python -m scripts.launch_final_demo --rollback
```

Baseline and rollback SHA-256: `eb177b5824fddf3dc351c06f86de40b49d8385694ce36746031a6e4540298bc1`. Rollback copy: `/Users/davidbonakdar/Documents/GitHub/Project-Spectra/models/baseline-rollback-20261011.joblib`. The original `models/baseline.joblib` is unchanged. Stopping the launcher also removes its process-local model selection; ordinary app launch still uses the baseline by default.

## Checks and remaining limitations

36 focused tests passed with no failures or skips: one-time evaluation/replay safeguards, partial-attempt lockout, raw shard identity/hash checks, speaker leakage/cache rejection, selected checkpoint and rollback loading, exact inference contract, malformed/silent/missing/incompatible model handling, preprocessing parity, app adapter, spectrogram construction and Streamlit rendering. These mechanics tests use generated tones and are not additional speech benchmark evaluations. The full suite was not run; optimization was not resumed. One pre-existing Librosa deprecation warning remains.

The actual frozen-model launcher started a temporary localhost Streamlit server and returned HTTP 200 `ok` at its health endpoint; the temporary server was stopped. Initial sandbox socket restrictions were resolved for this check. Check evidence is in `/Users/davidbonakdar/Documents/GitHub/Project-Spectra/runs/inthewild-final-20261011/demo_checks.json`. There are no unresolved local demo blockers. The historical seal-audit and source/generalization limits above remain unresolved scientific limitations.
