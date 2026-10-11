# Backend: Gary Stafford v4 local detector

Use Python **3.11** (`.python-version`). From the repository root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-backend.txt
.venv/bin/python -m pytest -q tests/test_audio.py tests/test_features.py tests/test_dataset.py tests/test_model.py
```

`requirements-backend.txt` owns backend audio, classifier, download, and test dependencies. The
frontend can include it from its requirements file with
`-r requirements-backend.txt` when the teammates integrate. The classifier is
scikit-learn `StandardScaler` + `LogisticRegression`; core inference is offline.
Numba is pinned to 0.60.0 for Python 3.11 / NumPy 1.26 compatibility and
prebuilt macOS Intel wheels.

## Frontend interface

```python
from src.audio import AudioDecodeError, decode_audio, split_windows
from src.features import FEATURE_NAMES, extract_features
from src.spectrogram import log_mel_spectrogram

try:
    recording = decode_audio(uploaded_file.getvalue(), uploaded_file.name)
except AudioDecodeError as error:
    # Display str(error) as invalid audio.
    ...
else:
    windows = split_windows(recording.samples)
    mel_db = log_mel_spectrogram(windows[0].samples)
    features = extract_features(windows[0].samples)
```

- `decode_audio(bytes, filename=None)` returns an `AudioRecording` with
  `samples` (mono float32), `sample_rate` (16000), `duration_s`, and `is_silent`.
  Filename never selects a decoder or accesses a path. libsndfile detects file
  content; WAV and FLAC are tested. Other format support depends on libsndfile.
  Limits: 25 MiB upload, 120 seconds, 8 channels, source rate <=192 kHz.
  Empty, unsupported, malformed, and non-finite audio raise `AudioDecodeError`.
- `split_windows(samples)` returns non-overlapping four-second `AudioWindow`s.
  The final window is zero-padded to 64000 samples; `start_s`, `end_s`, and
  `valid_samples` exclude padding. Padding is included in analysis statistics,
  so future training and inference must both use this windowing policy.
- `log_mel_spectrogram(samples)` returns float32 `(64, frames)` dB values at
  16 kHz, FFT size 512, hop 160, constant boundary padding, and fixed unit-power
  reference. For a four-second window it is `(64, 401)`; silence is -100 dB.
  Plot columns at `frame_index * 160 / 16000` seconds and offset by window start.
  Row centers use `librosa.mel_frequencies(n_mels=66, fmin=0, fmax=8000)[1:-1]`.
- `extract_features(samples)` returns 52 float32 values in `FEATURE_NAMES` order:
  20 MFCC means, 20 MFCC standard deviations, then mean/std for centroid,
  bandwidth, 85% rolloff, flatness, zero crossing rate, and RMS. No fitting or
  scaling occurs. Analysis functions require nonempty finite mono 16 kHz inputs.

Silence (RMS <=1e-5) is flagged, not assigned an authenticity label.

## Dataset and explicit workflow

Credit: Gary Stafford, [Deepfake Audio Detection Dataset (v4)](https://huggingface.co/datasets/garystafford/deepfake-audio-detection),
CC BY 4.0. It contains 1,866 English mono 16 kHz FLAC clips: 933 `real/` (label
0), 933 `fake/` (label 1). Hugging Face exposes **one `train` split**, with no
official validation or test partition. We use raw folders, never Parquet, and
never infer labels from alphabetical order. No dataset or model is bundled.

From repository root, after installing backend requirements above:

```sh
.venv/bin/python -m scripts.download_dataset
.venv/bin/python -m scripts.prepare_dataset --seed 42
.venv/bin/python -m scripts.train_model
.venv/bin/python -m scripts.evaluate_model
.venv/bin/python -m scripts.infer_audio --test-clip
```

The download is substantial, only runs on explicit command,
and requests only `real/*.flac`, `fake/*.flac`, `README.md`. It pins repository
revision `fcf5344bb7f82b54b6b932291326d29750ef1e82` (v4 inspected 2026-10-11).
`scripts/dataset_manifest.json` preserves all inspected original filenames.
Hub local-directory metadata permits resuming without also downloading Parquet
or keeping a second full Hub snapshot. `data/deepfake-audio-detection/provenance.json`
records credit/license/revision; the original dataset README is preserved.
Network failures stop with a nonzero exit code; rerun the same command.

Preparation checks the full manifest, FLAC format, rate/channels/duration, local
decoding, silence, and duplicate decoded waveforms. Missing, unexpected, empty,
corrupt, or duplicate clips stop preparation and are listed in
`data/preparation_report.json`; they are never silently omitted. The downloader's
`download_report.json` checks inventory only; preparation checks actual decoding.

Every `yt_0000_*` belongs to `yt_0000`, including `_c_` and `_p2_` chunks.
Synthetic recordings likewise use vendor prefix plus ID (`el_0001_*` belongs to
`el_0001`). All 1,866 inspected filenames match this rule: 14 real and 96 synthetic
source groups. Unknown patterns fail rather than falling back to a row split.
Seed 42 searches whole-group assignments using only class/sample counts, targeting
70/15/15 approximately; each partition must have both classes. `data/splits.csv`
persists original paths, labels, group/vendor IDs, membership, seed, file hashes,
and decoded-waveform hashes. Existing splits are preserved and verified against
data/seed; changed membership/data are rejected. No row-level random split is used.

Training extracts the existing 52 spectral/MFCC features from the same padded
four-second windows used by inference. Features cache in `data/features/` by
audio hash and feature policy. Only train/validation audio is processed during
training. The scaler and logistic regression fit only training windows; each
clip has unit total weight shared by its windows' actual durations. Clip scores
are the duration-weighted mean of segment scores. Threshold maximizes validation
clip balanced accuracy, with ties nearest 0.5. Validation metrics and individual
scores save to `runs/baseline/validation_metrics.json`.

`models/baseline.joblib` stores the fitted pipeline, threshold, exact feature/
window settings, training groups, seed, split hash, validation results, and
scikit-learn version. Only load locally generated trusted joblib artifacts
(joblib uses pickle). `data/`, `runs/`, and model joblib files are already gitignored.

Evaluation requires unchanged splits and the frozen training model. It accesses
test audio only in this explicit final step and saves measured clip metrics,
scores, groups, and model/split hashes to `runs/baseline/test_metrics.json`.
Repeated evaluation replays saved results without rescoring. A `test_started.json`
marker prevents automatic retries after partial test access. Inspect failed runs
before manual recovery; do not tune against exposed test results. Training refuses
to overwrite a model or reuse a run after final test access. Metrics report
accuracy, balanced accuracy, synthetic F1, ROC AUC, and a real/fake confusion matrix.

To screen your own local file after training:

```sh
.venv/bin/python -m scripts.infer_audio --audio /absolute/path/recording.wav
```

## Inference contract and limitations

```python
from src.inference import predict_audio
result = predict_audio(audio_bytes, filename=None)
```

This keeps exactly the seven AGENTS.md keys: `status`, `label`,
`synthetic_score`, `threshold`, `segments`, `model_version`, `message`.
No optional fields. Segment times exclude padding. `synthetic_score` is the
duration-weighted mean of model segment outputs, **not a calibrated probability**.
The raw logistic score is in [0,1]; no probability calibration was verified.
An absolute score distance <=0.05 from the validation threshold yields
`status="ok", label="inconclusive"`; this is a fixed abstention rule, not a verified
uncertainty interval. Evaluation binary metrics use the threshold without this
UI abstention rule.

Missing/incompatible models give `unavailable` with null scores; malformed audio
gives `invalid`. With an available model, silence or clips shorter than 0.25 s give
`invalid`. All error states are inconclusive with empty segments and an actionable
message. Before a model exists, decoded audio including silence gives `unavailable`.
Default artifact path is repository-relative `models/baseline.joblib`; a trusted
alternative can be set with `ECHOPROOF_MODEL_PATH` before invoking the API.
Nothing is downloaded or trained during imports or Streamlit startup. The existing
frontend adapter can call this exact API; frontend files were not changed.

This small English dataset may distinguish YouTube recording artifacts from clean
TTS rather than synthesis itself. Filename grouping recovers original source IDs,
but cannot prove speaker/text independence across different IDs/vendors. There
are only 14 real recording groups. Source-group test metrics do not establish
unseen-generator generalization. A separate vendor holdout and domain-shift check
have **not** been performed. Predictions are research estimates, not forensic proof.

Backend tests use generated tones in temporary directories only to verify pipeline
mechanics; their measurements are not speech benchmark evidence. Actual dataset
metrics/checkpoints exist only after the explicit download/train/evaluate commands.
