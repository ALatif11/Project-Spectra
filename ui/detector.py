"""Thin adapter over the backend contract; never fabricates predictions."""

import hashlib
import html
import os
from pathlib import Path

REQUIRED = {"status", "label", "synthetic_score", "threshold", "segments", "model_version", "message"}
LABELS = {
    "likely_human": "Model output: likely human",
    "likely_synthetic": "Model output: likely synthetic",
    "inconclusive": "Model output: inconclusive",
}


def _unavailable(message: str) -> dict:
    return {"status": "unavailable", "label": "inconclusive", "synthetic_score": None, "threshold": None,
            "segments": [], "model_version": None, "message": message}


def _model_path() -> Path:
    from src.model import MODEL_PATH
    return Path(os.environ.get("ECHOPROOF_MODEL_PATH") or MODEL_PATH)


def model_source() -> tuple[str, bool, bool]:
    """(display path, exists, overridden) for the checkpoint predict_audio will load; mirrors src/inference.py."""
    try:
        from src.model import MODEL_PATH
    except ImportError:
        return "unknown", False, False
    override = os.environ.get("ECHOPROOF_MODEL_PATH")
    path = _model_path()
    try:
        shown = path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        shown = path.name
    return shown, path.is_file(), bool(override)


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


def result_html(res: dict) -> str:
    """Markup for the detector card. Scores are shown raw; never as probabilities."""
    ok = res["status"] == "ok"
    title = LABELS.get(res["label"], LABELS["inconclusive"]) if ok else {"invalid": "Invalid audio"}.get(res["status"], "Detector unavailable")
    score, thr = res["synthetic_score"], res["threshold"]
    meter = ("" if not ok else
             f'<div class="meter" title="score vs threshold"><b style="width:{score * 100:.1f}%"></b>'
             f'<i style="left:{thr * 100:.1f}%"></i></div>')
    near = ('<div class="note">The score is close to the decision threshold, so the result is shown as inconclusive.</div>'
            if ok and res["label"] == "inconclusive" else "")
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    return (
        '<span class="pill warn">Experimental · not validated across diverse microphone recordings</span>'
        f'<div class="verdict {"" if ok else "off"}">{title}</div>'
        f'<div class="note">{html.escape(res["message"])}</div>'
        f'<div class="stats"><div class="stat"><b>{fmt(score)}</b><span>Model score (synthetic)</span></div>'
        f'<div class="stat"><b>{fmt(thr)}</b><span>Model decision threshold</span></div></div>'
        f'{meter}{near}'
        '<div class="note">Model scores are uncalibrated classifier outputs, not a confidence level or a verified '
        "probability that the audio is authentic or synthetic. Not forensic proof.</div>"
        '<p class="note subtle">Genuine speech may be scored as synthetic under unfamiliar recording conditions, '
        "such as different microphones, rooms, languages, or long pauses between words.</p>"
        + (f'<span class="pill">model version {html.escape(str(res["model_version"]))}</span>' if res["model_version"] else "")
    )


def model_html() -> str:
    shown, exists, overridden = model_source()
    kind = "selected via ECHOPROOF_MODEL_PATH" if overridden else "saved baseline (default)"
    if exists:
        digest = hashlib.sha256(_model_path().read_bytes()).hexdigest()[:12]
        return f'<span class="pill ok">{html.escape(kind)} · {html.escape(shown)} · sha256 {digest}</span>'
    return (f'<span class="pill warn">model file not found · {html.escape(shown)}</span>'
            '<p class="note">Copy the trained checkpoint into place. Nothing is trained or predicted until it exists.</p>')
