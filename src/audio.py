"""Bounded, in-memory audio decoding and consistent windowing."""

from dataclasses import dataclass
from io import BytesIO

import librosa
import numpy as np
from numpy.typing import NDArray
import soundfile as sf

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 120
MAX_CHANNELS = 8
MAX_SOURCE_SAMPLE_RATE = 192_000


class AudioDecodeError(ValueError):
    """The upload cannot be safely decoded as a supported recording."""


@dataclass(frozen=True)
class AudioRecording:
    samples: NDArray[np.float32]
    sample_rate: int = SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def is_silent(self) -> bool:
        """Flag near silence (RMS <= 1e-5); never infer authenticity."""
        return bool(np.sqrt(np.mean(self.samples.astype(np.float64) ** 2)) <= 1e-5)


@dataclass(frozen=True)
class AudioWindow:
    samples: NDArray[np.float32]
    start_s: float
    end_s: float
    valid_samples: int


def validate_samples(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Require a nonempty, finite mono waveform."""
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Audio must be a nonempty mono waveform.")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains non-finite samples.")
    return waveform


def decode_audio(audio_bytes: bytes, filename: str | None = None) -> AudioRecording:
    """Decode by content using libsndfile, downmix, and resample locally.

    Filename is informational only. No paths, remote APIs, or decoder subprocesses
    are used. Limits are checked before allocating the decoded waveform. Silence
    is valid audio, flagged on the returned recording. Unsupported/corrupt input
    raises AudioDecodeError so the caller can display an actionable error.
    """
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise AudioDecodeError("Upload a nonempty audio file (for example WAV or FLAC).")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise AudioDecodeError("Audio upload exceeds the 25 MiB limit.")
    try:
        with sf.SoundFile(BytesIO(audio_bytes)) as source:
            if not 1 <= source.channels <= MAX_CHANNELS:
                raise AudioDecodeError("Audio must have between 1 and 8 channels.")
            if not 1 <= source.samplerate <= MAX_SOURCE_SAMPLE_RATE:
                raise AudioDecodeError("Audio sample rate must be at most 192 kHz.")
            if source.frames <= 0:
                raise AudioDecodeError("Audio file contains no samples.")
            if source.frames > MAX_DURATION_SECONDS * source.samplerate:
                raise AudioDecodeError("Audio exceeds the 120-second duration limit.")
            source_rate = source.samplerate
            # Downmix in bounded blocks rather than allocating a full multichannel file.
            mono = np.empty(source.frames, dtype=np.float32)
            offset = 0
            while offset < len(mono):
                block = source.read(frames=min(source_rate, len(mono) - offset),
                                    dtype="float32", always_2d=True)
                if len(block) == 0:
                    raise AudioDecodeError("Audio file is truncated; upload a complete recording.")
                if not np.isfinite(block).all():
                    raise AudioDecodeError("Audio contains non-finite samples.")
                mono[offset : offset + len(block)] = block.mean(axis=1, dtype=np.float64)
                offset += len(block)
        if source_rate != SAMPLE_RATE:
            mono = librosa.resample(mono, orig_sr=source_rate, target_sr=SAMPLE_RATE)
        return AudioRecording(validate_samples(mono))
    except AudioDecodeError:
        raise
    except (RuntimeError, ValueError, OSError, OverflowError) as exc:
        raise AudioDecodeError("Cannot decode audio; upload a complete WAV or FLAC recording.") from exc


def split_windows(samples: NDArray[np.float32]) -> list[AudioWindow]:
    """Non-overlapping four-second windows at 16 kHz; zero-pad the last.

    Times describe real audio, excluding padding. Short recordings produce one
    window; exact multiples do not produce an extra empty window.
    """
    waveform = validate_samples(samples)
    window_size = int(WINDOW_SECONDS * SAMPLE_RATE)
    windows = []
    for start in range(0, len(waveform), window_size):
        chunk = waveform[start : start + window_size]
        padded = np.zeros(window_size, dtype=np.float32)
        padded[: len(chunk)] = chunk
        windows.append(AudioWindow(padded, start / SAMPLE_RATE,
                                   (start + len(chunk)) / SAMPLE_RATE, len(chunk)))
    return windows
