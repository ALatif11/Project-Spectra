import numpy as np
import pytest

from src.audio import SAMPLE_RATE, decode_audio, split_windows
from src.features import FEATURE_NAMES, extract_features
from src.spectrogram import log_mel_spectrogram
from test_audio import encode


@pytest.mark.parametrize("length", [1, 640, 64_000])
def test_silence_has_finite_features_and_spectrogram(length):
    samples = np.zeros(length, dtype=np.float32)
    mel = log_mel_spectrogram(samples)
    features = extract_features(samples)
    assert mel.shape[0] == 64
    assert np.isfinite(mel).all()
    np.testing.assert_allclose(mel, -100, atol=1e-4)
    assert features.shape == (52,)
    assert np.isfinite(features).all()
    assert features[FEATURE_NAMES.index("rms_mean")] == 0
    assert features[FEATURE_NAMES.index("zcr_mean")] == 0


def test_pipeline_deterministic_and_frequency_sensitive():
    time = np.arange(64_000, dtype=np.float32) / SAMPLE_RATE
    audio_bytes = encode(0.5 * np.sin(2 * np.pi * 440 * time))
    window = split_windows(decode_audio(audio_bytes).samples)[0]
    first = extract_features(window.samples)
    second = extract_features(window.samples)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(log_mel_spectrogram(window.samples), log_mel_spectrogram(window.samples))
    assert first.dtype == np.float32
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == len(first)
    assert first[FEATURE_NAMES.index("centroid_mean")] == pytest.approx(440, abs=15)
    assert first[FEATURE_NAMES.index("rms_mean")] == pytest.approx(0.5 / np.sqrt(2), abs=0.01)
    higher = extract_features(0.5 * np.sin(2 * np.pi * 2000 * time))
    assert higher[FEATURE_NAMES.index("centroid_mean")] > first[FEATURE_NAMES.index("centroid_mean")]
    assert not np.array_equal(first, higher)
    assert log_mel_spectrogram(window.samples).shape == (64, 401)


@pytest.mark.parametrize("samples", [np.zeros(0), np.zeros((2, 2)), np.array([np.inf])])
@pytest.mark.parametrize("function", [extract_features, log_mel_spectrogram])
def test_analysis_rejects_invalid_waveforms(samples, function):
    with pytest.raises(ValueError):
        function(samples)
