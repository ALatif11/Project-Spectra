"""Thin adapter over the backend contract; never fabricates predictions."""

REQUIRED = {"status", "label", "synthetic_score", "threshold", "segments", "model_version", "message"}
LABELS = {
    "likely_human": "Likely human speech",
    "likely_synthetic": "Likely synthetic speech",
    "inconclusive": "Inconclusive",
}


def _unavailable(message: str) -> dict:
    return {"status": "unavailable", "label": "inconclusive", "synthetic_score": None, "threshold": None,
            "segments": [], "model_version": None, "message": message}


def run_detector(audio_bytes: bytes, filename: str | None) -> dict:
    """Call src.inference.predict_audio (local, no network). Falls back to `unavailable`."""
    try:
        from src.inference import predict_audio
    except ImportError:
        return _unavailable("Detector not available yet: src/inference.py has not been integrated.")
    try:
        result = predict_audio(audio_bytes, filename)
    except Exception as exc:  # contract says it should not raise; surface it honestly
        return _unavailable(f"Detector raised {type(exc).__name__}. Check the backend logs.")
    if not isinstance(result, dict) or not REQUIRED <= result.keys():
        return _unavailable("Detector returned a response that does not match the agreed schema.")
    return result
