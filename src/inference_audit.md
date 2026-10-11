# EchoProof inference diagnosis — 2026-10-11

**Conclusion: likely domain generalization failure, with no verified inference implementation bug.** The user-attested genuine English microphone recording `/Users/davidbonakdar/Downloads/count-to-five-in-fre.wav` reproduces a baseline synthetic score of **0.9988675517388323**. One example establishes a failure, not a population error rate. Neither checkpoint nor training history was changed; no retraining, paid calls, Luminal experiments, or MongoDB work was performed.

## Actual model and preprocessing

`app.py` calls `ui.detector.run_detector`, which calls `src.inference.predict_audio`. The default is `models/baseline.joblib`, unless the launching process sets `ECHOPROOF_MODEL_PATH`. The audit environment has no override, and no running Streamlit process was found, so an earlier app process's environment cannot be independently recovered. The baseline directly reproduces the reported rounded 0.999 result.

| Artifact | SHA-256 | Model | Validation-selected threshold |
|---|---|---|---:|
| `models/baseline.joblib` | `eb177b5824fddf3dc351c06f86de40b49d8385694ce36746031a6e4540298bc1` | StandardScaler → logistic regression, C=1, seed 42 | 0.33173498266474855 |
| `models/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e.joblib` | `5f19ce37e01ecdace2ebad6193b6296b1314cb7c5dbb12abea2015174fb68ecf` | StandardScaler → logistic regression, C=0.1, seed 42 | 0.39278538241090233 |

Both fitted pipelines have `classes_ == [0, 1]` and 52 inputs. Column 1 is therefore synthetic. The model loader rejects another class ordering rather than silently interpreting it. Directory labels and the downloaded dataset card independently agree: real=0, fake=1. The UI displays that same synthetic score and explicitly calls it an uncalibrated model output.

Training and inference share `decode_audio`, `split_windows`, `window_features`, and the ordered `FEATURE_NAMES`: local SoundFile float32 decoding, arithmetic channel averaging, Librosa resampling to 16 kHz, **no peak/RMS loudness normalization**, non-overlapping four-second windows, final zero padding. The 52 values are 20 MFCC means then 20 MFCC standard deviations, followed by mean/std pairs for centroid, bandwidth, rolloff, flatness, zero-crossing rate, and RMS. Log-Mel uses FFT 512, hop 160, 64 bands, power 2, centered constant padding, 0–8 kHz, dB reference 1, floor 1e-10, top_db 80. The fitted scaler is embedded in each pipeline; inference neither substitutes nor refits it.

The recording is 48 kHz mono PCM16, 7.3386875 seconds after resampling. Its windows score **0.9999988079** and **0.9975122213**, weighted by **4.0** and **3.3386875** seconds. Aggregation is the duration-weighted mean of window scores, not the maximum or a sigmoid of the mean features.

Matched PCM16 WAV and FLAC conversions of this exact recording have **zero waveform difference, zero feature difference, and identical scores**. Four original training/validation examples, one from each class in each partition, have freshly extracted versus historical cached feature differences at most **3.8147e-6**, with identical window weights. These are tiny numerical differences, not bitwise historical reproducibility. Feature ordering/settings and dimensions match the artifact. Reconstructing baseline scaler moments from all 1,148 training clips gives mean error ≤1.95e-7 and variance relative error ≤1.40e-9; train clip weights total 1,148. The alternative checkpoint's scaler differs from baseline only at approximately 1e-15 relative precision. Its microphone score is also wrong at **0.9852940**; switching checkpoints is not a remedy.

## Split and source audit

The saved manifest hash matches both checkpoints: `01386211f92c06e24807c77345c108be0689a38292be75b7e2081ab2471da4da`, seed 42. All 1,866 filenames validate against their directory labels and inspected source patterns. `_p2` and `_c` variants retain the same original source identity. Duplicate-linked source components do not cross partitions. There are **216 excluded normalized-waveform duplicates**, leaving **1,650** clips:

| Partition | Genuine | Synthetic | Genuine recording sources |
|---|---:|---:|---|
| Train | 645 | 503 | 8 YouTube sources |
| Validation | 142 | 108 | 3 YouTube sources |
| Test | 143 | 109 | 3 YouTube sources |

Filename source grouping and recorded exact PCM duplicate grouping are verifiable. The historical PCM inventory was not fully re-decoded in this audit; source-byte hashes were checked for all training clips consumed by the diagnostics. Shared speakers, text, edited overlapping content, near duplicates, and vendor provenance cannot be independently established from filenames.

All **211 Speechify-labelled clips** were excluded as exact PCM duplicates of Luvvoice/Polly-labelled clips (125/86). Thus six advertised vendor labels do not constitute six independent retained vendor sources. All retained test vendor prefixes also occur in training; this is not an unseen-vendor evaluation.

The manifest records all 933 original real clips as **44.1 kHz stereo**, and all 933 fake clips as **16 kHz mono**, despite the card's blanket 16 kHz mono description. Local resampling/downmixing is consistent, but differences in recording/encoding history can remain. With only 14 real YouTube recordings and no real microphone training examples, source artifacts are a substantial confound.

## Why this genuine recording scores so highly

Against real training windows, the upload has **3 of 52 features outside the observed range in window 1**, and **1 in window 2**. Window 1 RMS variability is **0.113757**, versus real-training mean **0.039483**, standard deviation **0.017076**, maximum **0.087424**: approximately **4.35 standard deviations above the mean**. MFCC 11 and 14 variability also exceed the real-training range; window 2 rolloff variability exceeds it.

Its logistic decision values are approximately **13.69** and **5.99**, explaining saturated synthetic outputs. Loudness variation and multiple MFCC/spectral terms push the decision toward synthetic. These correlated contributions are descriptive evidence of the classifier's behavior, not proof that any individual feature causes the failure or detects synthesis.

The source contains pauses (163 near-silent 10 ms frames using RMS ≤1e-5), and five samples reach the negative PCM boundary. These observations do not establish causal effects. Full-upload silence is rejected, but silent windows within audible recordings are scored and receive their full duration weight. The dataset was previously segmented at speech boundaries, whereas arbitrary microphone uploads are not. This is a duration/speech-content domain mismatch even though the backend's training/inference window code is shared.

Short audible uploads down to 0.25 seconds are accepted and padded to four seconds, outside the dataset's 2.5–13 second clip coverage; long uploads are capped at 120 seconds and share the same window policy. Stereo antiphase cancellation, long pauses, clipping, level, microphone response, room noise, and recording processing can change model features. Do not change normalization or segmentation only at inference: any policy change requires consistent feature regeneration and training in a new artifact.

## Errors and score interpretation

From preserved final reports, without rerunning or tuning the final test:

| Model | Balanced accuracy | False positives (genuine → synthetic) | False negatives (synthetic → genuine) |
|---|---:|---:|---:|
| Baseline | 96.63501636% | 7/143 = 4.90% | 2/109 = 1.83% |
| Alternative | 96.41688587% | 5/143 = 3.50% | 4/109 = 3.67% |

These are threshold decisions before the UI abstention rule. Six of the baseline's seven genuine errors belong to `yt_0003`, and one to `yt_0006`; source clustering further limits interpretation of clip-level accuracy. Some threshold errors can appear inconclusive in the UI.

The separate external diagnostic set contains **one user-attested genuine English microphone recording; one fails** both the baseline threshold and UI verdict. It contains **zero external synthetic recordings**, so external false-negative behavior is unmeasured. Do not combine this 1/1 diagnostic observation with dataset metrics or treat it as a reliable microphone error-rate estimate.

There is no saved calibration validation or fitted calibrator. **0.999 is not 99.9% confidence that the recording is synthetic.** Even a logistic-regression `predict_proba` output needs representative calibration evidence; the duration-weighted aggregation adds another reason not to assume recording-level calibration. See [scikit-learn's calibration documentation](https://scikit-learn.org/stable/modules/calibration.html). The existing ±0.05 abstention band is explicitly a heuristic; it cannot protect against a confidently wrong domain-shift result.

## Quickest defensible hackathon response

1. Immediately disclose the microphone failure and unvalidated recording conditions in the demo; retain the raw score without presenting authenticity as settled. Use documented source-held-out examples to demonstrate the existing research pipeline. Treat arbitrary microphone screening as inconclusive for operational decisions until target-domain validation exists.
2. Freeze this recording and future diagnostic examples in a separately versioned **external evaluation** manifest; never put them into the retraining set after observing their results. Collect separate genuine examples across speakers, microphones/phones, rooms, gain levels, pauses, and recording codecs. Include independently sourced synthetic voices/vendors, both direct and re-recorded through the same devices, to weaken label/channel confounding. Obtain consent and keep private audio local.
3. After diagnosis approval, fit a new, small StandardScaler + logistic-regression artifact using original training plus new development examples, with speaker/session/source-aware train/validation grouping and controlled per-source weights. Regenerate features consistently if adding speech-aware segmentation or level normalization. Keep the original final test frozen; select a new threshold and any abstention/OOD policy only on a new representative development validation set with an explicit false-positive target. Evaluate the frozen external set once after freezing the new model. Report FPR/FNR separately and group uncertainty by speaker/session.
4. Check calibration on separate representative data before calling scores confidence; calibrate using group-disjoint development data if warranted. No arbitrary new score cutoff, recording exception, label inversion, or additional tuning on the old final test is justified.

## Files and integration

Added `scripts/audit_inference.py` (read-only reproducible audit), `tests/test_inference_audit.py` (format/resampling, class semantics, cache parity, aggregation, and diagnostic immutability), this report, and ignored local evidence `data/diagnostics/2026-10-11-inference-audit.json`. No production inference correction is warranted by the measurements. The audit refuses overwriting output or regenerating historical caches, never fits, and never evaluates the final test.

Future frontend integration belongs to `app.py`/`ui/` and should clarify microphone applicability; those teammate-owned files were not edited. A future validated preprocessing/retraining change would involve `src/audio.py`, `src/model.py`, `src/features.py`, and `src/training.py` as needed, plus new model/data artifacts; it is not implemented here.

Reproduce locally:

```sh
.venv/bin/python -m scripts.audit_inference \
  --real /Users/davidbonakdar/Downloads/count-to-five-in-fre.wav \
  --output /tmp/new-inference-audit.json
```

Use repeated `--real`/`--synthetic` arguments for independently labelled external samples. Generated regression-test waveforms test mechanics only, not speech detection performance.

## Verification results

- Targeted audio/features/model/audit suite: **54 passed**. New audit regressions account for nine tests.
- Full ordinary suite: **156 passed, 1 pre-existing frontend smoke failure**. `test_detector_unavailable_without_backend_inference` assumes no local checkpoint, but the real baseline exists and silence correctly returns `invalid`. This same failure was documented before this task in `src/deployment.md`; the teammate-owned test was not changed. With an explicitly missing checkpoint, all three frontend smoke tests pass.
- One existing Librosa deprecation warning; no new inference errors.
- An initial new format regression incorrectly encoded float samples independently into WAV/FLAC, whose PCM rounding differed by one least-significant bit. The test was corrected to quantize once and compare identical PCM, matching the actual recording experiment; all six corrected cases pass. No production decoder change was needed.
- All **14** pre-existing model/history/manifest files captured before the audit have unchanged SHA-256 hashes. Historical feature caches were read without regeneration. `git diff --check` passed; the pre-existing `.DS_Store` change was left alone.
