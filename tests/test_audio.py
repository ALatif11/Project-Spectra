from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from src.audio import AudioDecodeError, SAMPLE_RATE, decode_audio, split_windows


def encode(samples, rate=16_000, *, format="WAV", subtype="FLOAT"):
    buffer = BytesIO()
    sf.write(buffer, samples, rate, format=format, subtype=subtype)
    return buffer.getvalue()


@pytest.mark.parametrize("rate", [8_000, 16_000, 44_100, 48_000])
def test_valid_stereo_audio_resampled_to_mono(rate):
    time = np.arange(rate) / rate
    left = 0.3 * np.sin(2 * np.pi * 440 * time)
    recording = decode_audio(encode(np.column_stack((left, left * 0.5)), rate), "wrong.txt")
    assert recording.sample_rate == SAMPLE_RATE
    assert recording.samples.shape == (SAMPLE_RATE,)
    assert recording.samples.dtype == np.float32
    assert recording.duration_s == 1.0
    assert not recording.is_silent
    expected = 0.225 * np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
    np.testing.assert_allclose(recording.samples[100:-100], expected[100:-100], atol=0.002)


def test_flac_decodes_by_content():
    recording = decode_audio(encode(np.full(1000, 0.25), format="FLAC", subtype="PCM_16"))
    np.testing.assert_allclose(recording.samples, 0.25)


def test_silence_is_valid_and_flagged():
    recording = decode_audio(encode(np.zeros(SAMPLE_RATE)))
    assert recording.is_silent
    assert recording.duration_s == 1
    np.testing.assert_array_equal(recording.samples, 0)


@pytest.mark.parametrize("payload", [b"", b"not audio", b"RIFFbroken", None])
def test_invalid_files_raise_actionable_error(payload):
    with pytest.raises(AudioDecodeError, match="Upload|Cannot decode"):
        decode_audio(payload)


def test_empty_audio_file_rejected():
    with pytest.raises(AudioDecodeError, match="no samples"):
        decode_audio(encode(np.zeros(0)))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_float_audio_rejected(value):
    with pytest.raises(AudioDecodeError, match="non-finite"):
        decode_audio(encode(np.array([0, value], dtype=np.float32)))


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr("src.audio.MAX_UPLOAD_BYTES", 10)
    with pytest.raises(AudioDecodeError, match="25 MiB"):
        decode_audio(b"x" * 11)


def test_duration_limit(monkeypatch):
    monkeypatch.setattr("src.audio.MAX_DURATION_SECONDS", 1)
    with pytest.raises(AudioDecodeError, match="duration limit"):
        decode_audio(encode(np.zeros(SAMPLE_RATE + 1)))


def test_channel_limit():
    with pytest.raises(AudioDecodeError, match="channels"):
        decode_audio(encode(np.zeros((10, 9))))


def test_sample_rate_limit():
    with pytest.raises(AudioDecodeError, match="sample rate"):
        decode_audio(encode(np.zeros(10), rate=192_001))


@pytest.mark.parametrize("length, count", [(1, 1), (64_000, 1), (128_000, 2), (144_000, 3)])
def test_windows_preserve_samples_and_real_timestamps(length, count):
    samples = np.arange(length, dtype=np.float32) / np.float32(max(length, 1))
    windows = split_windows(samples)
    assert len(windows) == count
    restored = np.concatenate([window.samples[: window.valid_samples] for window in windows])
    np.testing.assert_array_equal(restored, samples)
    assert windows[-1].end_s == length / SAMPLE_RATE
    for i, window in enumerate(windows):
        assert window.samples.shape == (64_000,)
        assert window.start_s == i * 4
        assert window.end_s - window.start_s == window.valid_samples / SAMPLE_RATE
        np.testing.assert_array_equal(window.samples[window.valid_samples :], 0)
    windows[0].samples[0] = -1
    assert samples[0] == 0  # Windows own their buffers.


@pytest.mark.parametrize("samples", [np.zeros(0), np.zeros((2, 2)), np.array([np.nan])])
def test_windows_reject_invalid_waveforms(samples):
    with pytest.raises(ValueError):
        split_windows(samples)
