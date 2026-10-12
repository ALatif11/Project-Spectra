"""Thin adapter over the backend contract plus the result card markup; never fabricates predictions."""

import hashlib
import html
import os
from pathlib import Path

from ui.explain import level_words
from ui.theme import term, tip

REQUIRED = {"status", "label", "synthetic_score", "threshold", "segments", "model_version", "message"}
HEADLINES = {
    "likely_synthetic": ("ai", "This clip sounds closer to AI-generated speech"),
    "likely_human": ("human", "This clip sounds closer to a real human voice"),
    "inconclusive": ("unsure", "We can't tell from this clip"),
}
BAND = 0.05  # mirrors the abstention band in src/inference.py; display only

SCORE_TIP = ("A number from 0 to 1 that the model produces. Near 0: the audio resembles the real speech it was "
             "trained on. Near 1: it resembles AI-generated speech. It is not a percent chance.")
THRESHOLD_TIP = ("The cut-off the team picked using practice data. At or above it the clip is labeled AI-like, "
                 "below it human-like. Scores very close to it are called can't tell.")
WHY_TIP = ("Descriptive linear contributions: how far each measurement of this recording sits from the training "
           "average, weighted by the model. They describe the recording, not the person. They are not forensic causes "
           "and not an exact breakdown of the duration-averaged score.")
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


def checkpoint_label() -> str:
    shown, exists, _ = model_source()
    if not exists:
        return f"{shown} (not found)"
    return f"{shown} · sha256 {hashlib.sha256(_model_path().read_bytes()).hexdigest()[:12]}"


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


def headline(res: dict) -> tuple[str, str]:
    """(tone, sentence) for the top of the result card."""
    if res["status"] == "ok":
        return HEADLINES.get(res["label"], HEADLINES["inconclusive"])
    return "off", "We couldn't check this clip" if res["status"] == "invalid" else "Checking isn't available right now"


def meter_html(score: float, thr: float) -> str:
    lo, hi = max(0.0, thr - BAND), min(1.0, thr + BAND)
    return (f'<div class="meter" role="img" aria-label="Model score {score:.3f} on a scale from 0 (human-like) to 1 '
            f'(AI-like); threshold {thr:.3f}"><div class="track">'
            f'<span class="band" style="left:{lo:.1%};width:{hi - lo:.1%}"></span>'
            f'<span class="thr" style="left:{thr:.1%}"></span><span class="pin" style="left:{score:.1%}"></span></div>'
            '<div class="ends"><span>Human-like</span><span>AI-like</span></div></div>')


def why_html(why: list[dict] | None) -> str:
    if why is None:
        return ('<div class="section-h">Why</div><p class="note">The current model doesn\'t expose which measurements '
                "drove its score, so no breakdown is shown.</p>")
    items = "".join(
        f'<li class="{w["toward"]}"><span class="what">{html.escape(w["label"].capitalize())}'
        f'<span class="lvl"> · {level_words(w["level"])}</span></span>'
        f'<span class="dir {w["toward"]}">{"→ AI-like" if w["toward"] == "ai" else "→ human-like"}</span>'
        f'<span class="bar"><b style="width:{w["strength"]:.0%}"></b></span></li>' for w in why)
    return f'<div class="section-h">Why {tip(WHY_TIP)}</div><ul class="why">{items}</ul>'


def result_html(res: dict, why: list[dict] | None = None, warnings: list[str] | None = None) -> str:
    """Result card. Scores are shown raw and never as probabilities; wording follows the backend label."""
    ok = res["status"] == "ok"
    tone, sentence = headline(res)
    score, thr = res["synthetic_score"], res["threshold"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    sub = ("A signal to weigh, not proof. Whatever the result, verify before you act."
           if ok else html.escape(res["message"]))
    near = ('<p class="note">The score is close to the decision threshold, so the result is shown as inconclusive.</p>'
            if ok and res["label"] == "inconclusive" else "")
    warn_items = list(warnings or []) + (["Genuine speech may be scored as synthetic under unfamiliar recording "
                                          "conditions, such as different microphones, rooms, languages, or long "
                                          "pauses between words."] if ok else [])
    warns = ('<div class="section-h">Things that make this clip harder to judge</div><ul class="warns">'
             + "".join(f"<li>{html.escape(w)}</li>" for w in warn_items) + "</ul>") if warn_items else ""
    proof = term("Not forensic proof", "It cannot be used to prove a recording is real or fake, for example in a "
                                       "dispute or in court.")
    return (
        f'<div class="answer {tone}"><span class="dot"></span><div><h2>{sentence}</h2><p>{sub}</p></div></div>'
        + (meter_html(score, thr) if ok else "")
        + (f'<div class="figures"><span>Model score (synthetic) <b>{fmt(score)}</b> {tip(SCORE_TIP)}</span>'
           f'<span>Model decision threshold <b>{fmt(thr)}</b> {tip(THRESHOLD_TIP, "end")}</span></div>' if ok else "")
        + near
        + (why_html(why) if ok else "")
        + warns
        + '<div class="fine"><span class="chip warn" tabindex="0" data-tip="This detector is a research prototype. It '
          'has not been tested on everyday microphones, rooms or languages.">Experimental · not validated across '
          "diverse microphone recordings</span><br>Model scores are uncalibrated classifier outputs, not a confidence "
          f"level or a verified probability that the audio is authentic or synthetic. {proof}."
          + (f' <span data-tip="{html.escape(CHECKPOINT_TIP)}" tabindex="0">model version '
             f'{html.escape(str(res["model_version"]))}</span>' if res["model_version"] else "")
          + "</div>"
    )


def model_html() -> str:
    shown, exists, overridden = model_source()
    kind = "custom model (ECHOPROOF_MODEL_PATH)" if overridden else "saved baseline"
    if exists:
        return (f'<span class="chip ok" tabindex="0" data-tip="{html.escape(CHECKPOINT_TIP)}">{html.escape(kind)} · '
                f'{html.escape(checkpoint_label())}</span>')
    return (f'<span class="chip warn">model file not found · {html.escape(shown)}</span>'
            '<p class="note">Copy the trained checkpoint into place. Nothing is trained or predicted until it exists.</p>')
