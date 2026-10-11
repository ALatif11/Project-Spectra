"""Consistent log-Mel representation for local 16 kHz waveforms."""

import librosa
import numpy as np
from numpy.typing import NDArray

from src.audio import SAMPLE_RATE, validate_samples

N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64


def log_mel_spectrogram(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return (64 Mel bands, frames) in dB relative to unit power.

    Uses a 32 ms FFT, 10 ms hop and constant boundary padding. Fixed reference
    power preserves loudness differences; these values are not model scores.
    Silence remains finite. Inputs shorter than an FFT are padded explicitly.
    """
    waveform = validate_samples(samples)
    if waveform.size < N_FFT:
        waveform = np.pad(waveform, (0, N_FFT - waveform.size))
    power = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0.0, fmax=SAMPLE_RATE / 2,
        power=2.0, center=True, pad_mode="constant",
    )
    return librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=80.0).astype(np.float32)
