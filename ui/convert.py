"""Local M4A/AAC → WAV conversion so the backend decoder (libsndfile) can read it.

Conversion keeps the original sample rate and channels (16-bit PCM WAV); the backend still does its own
downmix and resampling, so preprocessing is unchanged. Runs on this computer with PyAV's bundled decoder
(optional dependency: pip install av). MP3, WAV, FLAC and OGG need no conversion.
"""

import io

import numpy as np
import soundfile as sf

CONVERT_EXT = (".m4a", ".mp4", ".aac")
MAX_BYTES = 25 * 1024 * 1024  # mirrors src/audio.py MAX_UPLOAD_BYTES
MAX_SECONDS = 120  # mirrors src/audio.py MAX_DURATION_SECONDS


class ConversionError(ValueError):
    """The file could not be converted; the message is safe to show to the user."""


def needs_conversion(filename: str | None) -> bool:
    return bool(filename) and filename.lower().endswith(CONVERT_EXT)


def to_wav(data: bytes) -> bytes:
    if len(data) > MAX_BYTES:
        raise ConversionError("The file is larger than 25 MiB.")
    try:
        import av
    except ImportError as exc:
        raise ConversionError("M4A support needs the optional 'av' package on this computer (pip install av).") from exc
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ConversionError("This file has no audio track.")
            chunks, rate, total = [], None, 0
            for frame in container.decode(stream):
                raw = frame.to_ndarray()
                arr = raw.astype(np.float32)
                if frame.format.is_planar:  # (channels, samples)
                    arr = arr.T
                else:  # packed: (1, samples*channels)
                    arr = arr.reshape(-1, len(frame.layout.channels))
                if np.issubdtype(raw.dtype, np.integer):
                    arr = arr / float(np.iinfo(raw.dtype).max)
                rate = frame.sample_rate
                chunks.append(arr)
                total += len(arr)
                if total > (MAX_SECONDS + 1) * rate:
                    raise ConversionError("The clip is longer than 120 seconds.")
    except ConversionError:
        raise
    except Exception as exc:  # corrupt or unsupported container
        raise ConversionError("This M4A file could not be read. Try exporting it again or use WAV/MP3.") from exc
    if not chunks or not rate:
        raise ConversionError("This file contains no audio.")
    out = io.BytesIO()
    sf.write(out, np.clip(np.concatenate(chunks), -1, 1), rate, format="WAV", subtype="PCM_16")
    return out.getvalue()
