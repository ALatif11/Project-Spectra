"""Deterministic MFCC and spectral statistics; no fitted transforms."""

import librosa
import numpy as np
from numpy.typing import NDArray

from src.audio import SAMPLE_RATE, validate_samples
from src.spectrogram import HOP_LENGTH, N_FFT, log_mel_spectrogram

N_MFCC = 20
SPECTRAL_NAMES = ("centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms")
FEATURE_NAMES = tuple(
    f"mfcc_{i:02d}_{stat}" for stat in ("mean", "std") for i in range(N_MFCC)
) + tuple(f"{name}_{stat}" for name in SPECTRAL_NAMES for stat in ("mean", "std"))


def extract_features(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return 52 values in FEATURE_NAMES order, suitable for a future classifier.

    Input is mono 16 kHz audio, normally one padded window from split_windows.
    MFCC means/stds (20 each) precede means/stds of six spectral descriptors.
    No randomness, data-dependent fitting, or authenticity estimate is involved.
    """
    waveform = validate_samples(samples)
    if waveform.size < N_FFT:
        waveform = np.pad(waveform, (0, N_FFT - waveform.size))
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram(waveform), n_mfcc=N_MFCC)
    magnitude = np.abs(librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                    center=True, pad_mode="constant"))
    descriptors = (
        librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE),
        librosa.feature.spectral_bandwidth(S=magnitude, sr=SAMPLE_RATE),
        librosa.feature.spectral_rolloff(S=magnitude, sr=SAMPLE_RATE, roll_percent=0.85),
        librosa.feature.spectral_flatness(S=magnitude),
        librosa.feature.zero_crossing_rate(waveform, frame_length=N_FFT, hop_length=HOP_LENGTH),
        librosa.feature.rms(y=waveform, frame_length=N_FFT, hop_length=HOP_LENGTH,
                            center=True, pad_mode="constant"),
    )
    values = [*mfcc.mean(axis=1), *mfcc.std(axis=1)]
    for descriptor in descriptors:
        values.extend((float(descriptor.mean()), float(descriptor.std())))
    features = np.asarray(values, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced non-finite values.")
    return features
