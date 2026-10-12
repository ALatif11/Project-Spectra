import io
from pathlib import Path

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest


def test_app_renders_without_upload():
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Check a clip", "How it works", "Model", "Agent experiments", "Results over time"]


def test_detector_unavailable_without_backend_inference(monkeypatch, tmp_path):
    from ui.detector import run_detector
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(tmp_path / "missing.joblib"))
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    res = run_detector(buf.getvalue(), "x.wav")
    assert res["status"] == "unavailable" and res["synthetic_score"] is None and res["segments"] == []


def test_figures_build_for_tone():
    from src.audio import decode_audio
    from ui import viz
    buf = io.BytesIO()
    t = np.arange(16000 * 5) / 16000
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000, format="WAV")
    rec = decode_audio(buf.getvalue(), "tone.wav")
    assert viz.waveform_fig(rec).data and viz.spectrogram_fig(rec).data


def _wav(seconds=1.0, amp=0.3):
    buf = io.BytesIO()
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(buf, (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000, format="WAV")
    return buf.getvalue()


def test_real_backend_without_model_is_unavailable(monkeypatch, tmp_path):
    from ui.detector import run_detector
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(tmp_path / "missing.joblib"))
    res = run_detector(_wav(), "tone.wav")
    assert res["status"] == "unavailable" and res["synthetic_score"] is None and res["label"] == "inconclusive"


def test_invalid_audio_passes_through():
    from ui.detector import run_detector
    assert run_detector(b"not audio", "x.wav")["status"] == "invalid"


def test_malformed_or_raising_backend_is_unavailable(monkeypatch):
    import src.inference as inf
    from ui.detector import run_detector
    monkeypatch.setattr(inf, "predict_audio", lambda b, f=None: {"status": "ok"})
    assert run_detector(b"x", None)["status"] == "unavailable"

    def boom(b, f=None):
        raise RuntimeError("x")
    monkeypatch.setattr(inf, "predict_audio", boom)
    assert run_detector(b"x", None)["status"] == "unavailable"


def _stub(label, score):
    return lambda b, f=None: {
        "status": "ok", "label": label, "synthetic_score": score, "threshold": 0.5,
        "segments": [{"start_s": 0.0, "end_s": 1.0, "synthetic_score": score}],
        "model_version": "test-stub", "message": "stub"}


def test_result_html_states():
    """Stub lives only in tests: checks rendering paths, not model behavior."""
    from ui.detector import result_html
    for label, text in [("likely_human", "closer to a real human voice"), ("likely_synthetic", "closer to AI-generated speech"),
                        ("inconclusive", "We can't tell")]:
        out = result_html(_stub(label, 0.52)(b"", None))
        assert text in out and "0.520" in out and "Model score (synthetic)" in out and "Not forensic proof" in out
        assert "not a confidence level or a verified probability" in out and "Experimental" in out
        assert "Genuine speech may be scored as synthetic" in out and "model version test-stub" in out
    assert "close to the decision threshold" in result_html(_stub("inconclusive", 0.52)(b"", None))
    out = result_html({"status": "unavailable", "label": "inconclusive", "synthetic_score": None, "threshold": None,
                       "segments": [], "model_version": None, "message": "<b>m</b>"})
    assert "Checking isn't available" in out and "&lt;b&gt;" in out and "meter" not in out and "Why" not in out


def test_model_source_reports_override(monkeypatch, tmp_path):
    from ui.detector import model_html, model_source
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(tmp_path / "missing.joblib"))
    assert model_source() == ("missing.joblib", False, True) and "model file not found" in model_html()
    monkeypatch.delenv("ECHOPROOF_MODEL_PATH")
    shown, _, overridden = model_source()
    assert shown == "models/baseline.joblib" and not overridden


def test_silent_audio_is_invalid_when_baseline_exists(monkeypatch):
    """Backend contract: with a model, silence is `invalid`; without one, everything is `unavailable`."""
    import pytest
    from src.model import MODEL_PATH
    from ui.detector import run_detector
    if not MODEL_PATH.is_file():
        pytest.skip("models/baseline.joblib not present on this machine")
    monkeypatch.delenv("ECHOPROOF_MODEL_PATH", raising=False)
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    res = run_detector(buf.getvalue(), "silence.wav")
    assert res["status"] == "invalid" and res["synthetic_score"] is None and res["segments"] == []


def test_why_and_warnings_render_without_inventing():
    from ui.detector import result_html
    why = [{"feature": "rms_std", "label": "how much the loudness changes", "toward": "ai", "strength": 1.0, "level": 2.1}]
    out = result_html(_stub("likely_synthetic", 0.9)(b"", None), why, ["Short clip."])
    assert "How much the loudness changes" in out and "much higher than typical" in out and "→ AI-like" in out
    assert "Short clip." in out
    assert "doesn't expose" in result_html(_stub("likely_human", 0.1)(b"", None), None)


def test_explain_helpers():
    from ui import explain
    assert explain.plain_name("mfcc_00_mean") == "overall energy of the voice"
    assert explain.plain_name("rms_std") == "how much the loudness changes"
    assert explain.level_words(0.2) == "about typical" and explain.level_words(-1) == "lower than typical"
    quiet = np.zeros(16000 * 2, dtype=np.float32)
    quiet[:4000] = 0.3
    warns = explain.quality_checks(quiet, 2.0)
    assert any("very short" in w for w in warns) and any("silence" in w for w in warns)
    md = explain.report_markdown("a.wav", "Sounds closer to AI", _stub("likely_synthetic", 0.9)(b"", None), None, [], "ckpt")
    assert "0.900" in md and "Not forensic proof" in md and "not uploaded" in md and "reportfraud.ftc.gov" in md


def test_factors_from_real_checkpoint_if_present(monkeypatch):
    import pytest
    from src.model import MODEL_PATH
    from ui import explain
    if not MODEL_PATH.is_file():
        pytest.skip("models/baseline.joblib not present on this machine")
    monkeypatch.delenv("ECHOPROOF_MODEL_PATH", raising=False)
    out = explain.factors(_wav(2.0), "tone.wav")
    assert out and len(out) == 3 and out[0]["strength"] == 1.0 and {o["toward"] for o in out} <= {"ai", "human"}


def test_m4a_converts_locally_and_mp3_needs_no_conversion():
    """Generated tone (mechanics only). M4A needs the optional 'av' package."""
    import pytest
    from ui import convert
    from src.audio import decode_audio
    assert convert.needs_conversion("Voice Memo.M4A") and not convert.needs_conversion("clip.mp3")
    av = pytest.importorskip("av")
    out = io.BytesIO()
    with av.open(out, "w", format="mp4") as c:
        s = c.add_stream("aac", rate=44100)
        t = np.arange(44100 * 2) / 44100
        frame = av.AudioFrame.from_ndarray((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None, :],
                                           format="flt", layout="mono")
        frame.sample_rate = 44100
        for packet in [*s.encode(frame), *s.encode(None)]:
            c.mux(packet)
    rec = decode_audio(convert.to_wav(out.getvalue()), "x.wav")
    assert 1.9 < rec.duration_s < 2.2
    with pytest.raises(convert.ConversionError):
        convert.to_wav(b"not audio")
