"""Thin adapter over the backend contract; never fabricates predictions."""

import hashlib
import html
import math
import os
from pathlib import Path

from ui.theme import term, tip

REQUIRED = {"status", "label", "synthetic_score", "threshold", "segments", "model_version", "message"}
LABELS = {
    "likely_human": "Model output: likely human",
    "likely_synthetic": "Model output: likely synthetic",
    "inconclusive": "Model output: inconclusive",
}
TONE = {"likely_human": "human", "likely_synthetic": "synth", "inconclusive": "mid"}
BAND = 0.05  # mirrors the abstention band in src/inference.py; display only

SCORE_TIP = ("A number from 0 to 1 that the model produces. Near 0: the audio resembles the real speech it was "
             "trained on. Near 1: it resembles the AI-generated speech. It is not a percent chance.")
THRESHOLD_TIP = ("The cut-off the team picked using practice data. Scores at or above it are labeled synthetic, "
                 "below it human. Scores very close to it are called inconclusive.")
GAUGE_TIP = ("The needle is the model score. Green: human-like. Red: synthetic-like. Amber: too close to call. "
             "The dark tick marks the decision threshold.")
EXPERIMENTAL_TIP = ("Trained on one public dataset of 14 real YouTube sources and several voice generators. "
                    "It has not been tested on everyday microphones, rooms, or languages.")
CHECKPOINT_TIP = ("The checkpoint is the saved, trained model file. The sha256 is its fingerprint: if it matches "
                  "the team's report, this app is running exactly that model.")


def _unavailable(message: str) -> dict:
    return {"status": "unavailable", "label": "inconclusive", "synthetic_score": None, "threshold": None,
            "segments": [], "model_version": None, "message": message}


def _model_path() -> Path:
    from src.model import MODEL_PATH
    return Path(os.environ.get("ECHOPROOF_MODEL_PATH") or MODEL_PATH)


def model_source() -> tuple[str, bool, bool]:
    """(display path, exists, overridden) for the checkpoint predict_audio will load; mirrors src/inference.py."""
    try:
        path = _model_path()
    except ImportError:
        return "unknown", False, False
    try:
        shown = path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        shown = path.name
    return shown, path.is_file(), bool(os.environ.get("ECHOPROOF_MODEL_PATH"))


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


def _pt(v: float, r: float = 120) -> tuple[float, float]:
    a = math.pi * (1 - v)
    return 150 + r * math.cos(a), 150 - r * math.sin(a)


def _arc(a: float, b: float, color: str) -> str:
    if b <= a:
        return ""
    (x1, y1), (x2, y2) = _pt(a), _pt(b)
    return (f'<path d="M{x1:.1f} {y1:.1f} A120 120 0 0 1 {x2:.1f} {y2:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="14"/>')


def gauge_svg(score: float, thr: float) -> str:
    lo, hi = max(0.0, thr - BAND), min(1.0, thr + BAND)
    (tx1, ty1), (tx2, ty2) = _pt(thr, 104), _pt(thr, 136)
    return (
        '<svg class="gauge" viewBox="0 0 300 172" role="img" '
        f'aria-label="Gauge: model score {score:.3f}, threshold {thr:.3f}">'
        + _arc(0, lo, "rgba(60,203,154,.6)") + _arc(lo, hi, "rgba(233,180,76,.65)") + _arc(hi, 1, "rgba(255,107,111,.6)")
        + f'<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="#ECEBF0" stroke-width="2.5"/>'
        f'<g class="needle" style="--a:{-90 + 180 * score:.1f}deg">'
        '<line x1="150" y1="150" x2="150" y2="46" stroke="#ECEBF0" stroke-width="3" stroke-linecap="round"/></g>'
        '<circle cx="150" cy="150" r="9" fill="#ECEBF0"/><circle cx="150" cy="150" r="3.5" fill="#15161B"/>'
        '<text x="22" y="170">0 · human-like</text><text x="278" y="170" text-anchor="end">synthetic-like · 1</text>'
        "</svg>"
    )


def result_html(res: dict) -> str:
    """Markup for the detector card. Scores are shown raw; never as probabilities."""
    ok = res["status"] == "ok"
    title = LABELS.get(res["label"], LABELS["inconclusive"]) if ok else {"invalid": "Invalid audio"}.get(res["status"], "Detector unavailable")
    tone = TONE.get(res["label"], "mid") if ok else "off"
    score, thr = res["synthetic_score"], res["threshold"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    gauge = (f'{gauge_svg(score, thr)}<p class="note dial">How to read the dial {tip(GAUGE_TIP, "below")}</p>'
             if ok else "")
    near = ('<p class="note">The score is close to the decision threshold, so the result is shown as inconclusive.</p>'
            if ok and res["label"] == "inconclusive" else "")
    version = (f'<span class="chip plain" data-tip="{html.escape(CHECKPOINT_TIP)}" tabindex="0">model version '
               f'{html.escape(str(res["model_version"]))}</span>') if res["model_version"] else ""
    proof = term("Not forensic proof", "It cannot be used to prove a recording is real or fake, "
                                       "for example in a dispute or in court.")
    return (
        f'<span class="chip warn" tabindex="0" data-tip="{html.escape(EXPERIMENTAL_TIP)}">'
        "Experimental · not validated across diverse microphone recordings</span>"
        f"{gauge}"
        f'<div class="verdict {tone}">{title}</div>'
        f'<p class="note">{html.escape(res["message"])}</p>'
        '<div class="readout">'
        f'<div><b>{fmt(score)}</b><span>Model score (synthetic) {tip(SCORE_TIP)}</span></div>'
        f'<div><b>{fmt(thr)}</b><span>Model decision threshold {tip(THRESHOLD_TIP, "end")}</span></div></div>'
        f"{near}"
        '<p class="note">Model scores are uncalibrated classifier outputs, not a confidence level or a verified '
        f"probability that the audio is authentic or synthetic. {proof}.</p>"
        '<p class="note subtle">Genuine speech may be scored as synthetic under unfamiliar recording conditions, '
        "such as different microphones, rooms, languages, or long pauses between words.</p>"
        f'<div class="model-tag">{version}</div>'
    )


def model_html() -> str:
    shown, exists, overridden = model_source()
    kind = "custom model (ECHOPROOF_MODEL_PATH)" if overridden else "saved baseline"
    if exists:
        digest = hashlib.sha256(_model_path().read_bytes()).hexdigest()[:12]
        return (f'<span class="chip ok" tabindex="0" data-tip="{html.escape(CHECKPOINT_TIP)}">{html.escape(kind)} · '
                f'{html.escape(shown)} · sha256 {digest}</span>')
    return (f'<span class="chip warn">model file not found · {html.escape(shown)}</span>'
            '<p class="note">Copy the trained checkpoint into place. Nothing is trained or predicted until it exists.</p>')
