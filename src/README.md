# Backend milestone 1

Use Python **3.11** (`.python-version`). From the repository root:

```sh
python3.11 -m venv /tmp/echoproof-venv
/tmp/echoproof-venv/bin/python -m pip install -r requirements-backend.txt
/tmp/echoproof-venv/bin/python -m pytest -q
```

`requirements-backend.txt` owns only backend audio and test dependencies. The
frontend can include it from its requirements file with
`-r requirements-backend.txt` when the teammates integrate. Librosa brings
scikit-learn transitively; no classifier, cloud SDK, or model is implemented.
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

Silence (RMS <=1e-5) is flagged, not assigned an authenticity label. No trained
model or `predict_audio` implementation exists in this milestone. The frontend
must continue showing `unavailable` for predictions with null scores and an
inconclusive label until a genuine model is integrated.
