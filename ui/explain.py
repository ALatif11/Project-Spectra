"""Plain-language "why", clip-quality checks and the downloadable report.

Everything here is computed directly from the loaded checkpoint and the decoded audio. Nothing is
generated or estimated by another model. Explanations exist only for checkpoints that expose linear
weights (scaler + logistic regression); other model types return None and the UI says so.
"""

from datetime import datetime

import numpy as np

_PLAIN = {
    "centroid": "brightness of the sound",
    "bandwidth": "range of pitches in the sound",
    "rolloff": "amount of high-pitched energy",
    "flatness": "background hiss or noise",
    "zcr": "sharp 's' and 'f' sounds",
    "rms": "loudness",
}


def plain_name(feature: str) -> str:
    """Everyday name for a feature such as mfcc_03_std or rms_mean."""
    base, _, stat = feature.rpartition("_")
    if base.startswith("mfcc_"):
        k = int(base.split("_")[1])
        what = "overall energy of the voice" if k == 0 else f"voice tone color (band {k})"
    else:
        what = _PLAIN.get(base, base.replace("_", " "))
    return f"how much the {what} changes" if stat == "std" else what


def factors(audio_bytes: bytes, filename: str | None, top: int = 3) -> list[dict] | None:
    """Top measured properties that moved the score, or None when this checkpoint can't be explained this way."""
    try:
        from src.audio import decode_audio
        from src.model import load_model, window_features
        from ui.detector import _model_path

        artifact = load_model(_model_path())
        steps = getattr(artifact["pipeline"], "named_steps", {})
        scaler, clf = steps.get("scaler"), steps.get("classifier")
        names = (artifact.get("feature_settings") or {}).get("feature_names")
        if scaler is None or not hasattr(clf, "coef_") or not names:
            return None
        feats, weights, _ = window_features(decode_audio(audio_bytes, filename))
        if feats.shape[1] != len(names) or clf.coef_.shape[1] != len(names):
            return None
        z = scaler.transform(feats)
        push = np.average(z * clf.coef_[0], axis=0, weights=weights)  # log-odds contribution per feature
        zbar = np.average(z, axis=0, weights=weights)
    except Exception:  # explanation is optional; never break the result card
        return None
    order = np.argsort(-np.abs(push))[:top]
    strongest = float(np.abs(push[order[0]])) or 1.0
    return [{"feature": names[i], "label": plain_name(names[i]), "toward": "ai" if push[i] > 0 else "human",
             "strength": float(abs(push[i]) / strongest), "level": float(zbar[i])} for i in order]


def level_words(z: float) -> str:
    if abs(z) < 0.5:
        return "about typical"
    return ("much " if abs(z) >= 2 else "") + ("higher" if z > 0 else "lower") + " than typical"


def quality_checks(samples: np.ndarray, duration_s: float) -> list[str]:
    """Factual reasons this clip may be harder to judge."""
    out = []
    if duration_s < 3:
        out.append("The clip is very short. Five seconds or more of one person speaking works best.")
    x = samples.astype(np.float64)
    frames = x[: len(x) // 400 * 400].reshape(-1, 400)  # 25 ms frames at 16 kHz
    if len(frames):
        level = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
        pauses = float((level < level.max() - 35).mean())
        if pauses > 0.4:
            out.append(f"About {pauses:.0%} of the clip is silence or pauses, which this model handles poorly.")
    if float((np.abs(x) >= 0.99).mean()) > 0.005:
        out.append("The audio is distorted (clipped), which can change the result.")
    return out


NEXT_STEPS = [
    ("Call them back on a number you already know.", "Hang up and dial the person directly, or reach them through family or friends."),
    ("Ask something only they would know.", "A family code word agreed in advance is the simplest check."),
    ("Treat payment requests as the warning sign.", "Gift cards, wire transfers, crypto or a cash courier are scam signs, whatever the voice sounds like."),
    ("Slow down.", "Pressure to act right now or keep it secret is a classic scam tactic."),
]
REPORT_LINKS = [("Report fraud (FTC)", "https://reportfraud.ftc.gov/"),
                ("FTC: scammers using AI voice cloning", "https://consumer.ftc.gov/node/80688")]


def report_markdown(filename: str, headline: str, res: dict, why: list[dict] | None, warnings: list[str],
                    checkpoint: str) -> str:
    score = "n/a" if res["synthetic_score"] is None else f"{res['synthetic_score']:.3f}"
    thr = "n/a" if res["threshold"] is None else f"{res['threshold']:.3f}"
    lines = [
        "# EchoProof check", "",
        f"- File: {filename}", f"- Checked: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Result: {headline}", f"- Model score: {score} (threshold {thr}; 0 = human-like, 1 = AI-like)",
        f"- Model: {res.get('model_version') or 'n/a'} · {checkpoint}", "",
    ]
    if why:
        lines += ["## Why", *[f"- {w['label'].capitalize()}: {level_words(w['level'])}; pushed toward "
                              f"{'AI-generated' if w['toward'] == 'ai' else 'human'}." for w in why], ""]
    if warnings:
        lines += ["## Things that make this clip harder to judge", *[f"- {w}" for w in warnings], ""]
    lines += ["## What to do next", *[f"- {t} {d}" for t, d in NEXT_STEPS],
              *[f"- {name}: {url}" for name, url in REPORT_LINKS], "",
              "Model scores are uncalibrated classifier outputs, not a confidence level or a verified probability "
              "that the audio is authentic or synthetic. Not forensic proof. The clip was analyzed on this computer "
              "and was not uploaded.", ""]
    return "\n".join(lines)
