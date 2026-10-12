"""Plotly figures: waveform and spectrogram from decoded audio, plus the backend's own segment scores."""

import librosa
import numpy as np
import plotly.graph_objects as go

from src.audio import SAMPLE_RATE, AudioRecording, split_windows
from src.spectrogram import HOP_LENGTH, N_MELS, log_mel_spectrogram
from ui.theme import TOKENS

T = TOKENS
INK, MUTE, SIGNAL, HUMAN, SYNTH, SCREEN = T["fg"], T["muted"], T["brand"], T["human"], T["ai"], T["bg"]
_FONT = "Geist, system-ui, sans-serif"
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8),
    font=dict(family=_FONT, color=MUTE, size=11),
    hoverlabel=dict(bgcolor=T["fg"], bordercolor=T["fg"], font=dict(family=_FONT, color=T["bg"], size=12)),
)
_GRID = dict(gridcolor="rgba(255,255,255,.05)", zeroline=False, linecolor=T["border"])
_SCREEN_GRID = dict(gridcolor="rgba(255,255,255,.05)", zeroline=False, linecolor=T["border"],
                    tickfont=dict(color=MUTE), title_font=dict(color=MUTE))


def waveform_fig(rec: AudioRecording, bins: int = 1600) -> go.Figure:
    x = rec.samples
    n = max(1, len(x) // bins)
    trim = x[: n * (len(x) // n)].reshape(-1, n)
    t = (np.arange(trim.shape[0]) * n + n / 2) / SAMPLE_RATE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=trim.max(1), line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=trim.min(1), fill="tonexty", fillcolor="rgba(91,140,255,.22)",
                             line=dict(color=SIGNAL, width=1),
                             hovertemplate="%{x:.2f} s into the clip<extra></extra>"))
    fig.update_layout(**_BASE, height=170, showlegend=False,
                      xaxis=dict(title="seconds", **_GRID), yaxis=dict(range=[-1, 1], showticklabels=False, **_GRID))
    return fig


def stitched_mel(rec: AudioRecording) -> tuple[np.ndarray, np.ndarray]:
    """(log-Mel, frame times) across the 4 s windows. Each window's frames are timed from that window's own
    start, the boundary frame shared with the next window is dropped, and zero padding is excluded."""
    windows = split_windows(rec.samples)
    parts, times = [], []
    for i, w in enumerate(windows):
        mel = log_mel_spectrogram(w.samples)
        t = w.start_s + np.arange(mel.shape[1]) * HOP_LENGTH / SAMPLE_RATE
        keep = t <= w.end_s + 1e-9 if i == len(windows) - 1 else t < w.end_s - 1e-9
        parts.append(mel[:, keep])
        times.append(t[keep])
    return np.hstack(parts), np.concatenate(times)


def spectrogram_fig(rec: AudioRecording) -> go.Figure:
    """Log-Mel (dB, fixed unit-power reference) stitched across the 4 s windows, padding removed."""
    mel, t = stitched_mel(rec)
    hz = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=SAMPLE_RATE / 2)[1:-1]
    fig = go.Figure(go.Heatmap(
        z=mel, x=t, y=hz, zmin=-80, zmax=float(max(mel.max(), -70)),
        colorscale=[[0, SCREEN], [.35, "#16213f"], [.65, "#3b63c9"], [.85, "#8fb3ff"], [1, "#f2f6ff"]],
        colorbar=dict(title=dict(text="loudness (dB)", side="right"), thickness=8, outlinewidth=0,
                      tickfont=dict(size=10, color=MUTE), title_font=dict(color=MUTE, size=10)),
        hovertemplate="at %{x:.2f} s, pitch around %{y:.0f} Hz was %{z:.0f} dB<extra></extra>",
    ))
    fig.update_layout(**{**_BASE, "paper_bgcolor": SCREEN, "plot_bgcolor": SCREEN, "margin": dict(l=14, r=10, t=14, b=12)},
                      height=300, xaxis=dict(title="seconds", **_SCREEN_GRID), yaxis=dict(title="pitch (Hz)", **_SCREEN_GRID))
    return fig


def segments_fig(segments: list[dict], threshold: float | None) -> go.Figure:
    """Backend segment scores; bars colored only by the backend's own threshold."""
    scores = [s["synthetic_score"] for s in segments]
    colors = [MUTE if threshold is None else (SYNTH if v >= threshold else HUMAN) for v in scores]
    fig = go.Figure(go.Bar(
        x=[(s["start_s"] + s["end_s"]) / 2 for s in segments],
        width=[s["end_s"] - s["start_s"] - 0.15 for s in segments],
        y=scores, marker=dict(color=colors, cornerradius=4),
        customdata=[[s["start_s"], s["end_s"]] for s in segments],
        hovertemplate="%{customdata[0]:.1f}–%{customdata[1]:.1f} s: model score %{y:.3f}<extra></extra>",
    ))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=INK, dash="dot", width=1),
                      annotation_text="threshold", annotation_font_color=INK, annotation_position="top left")
    fig.update_layout(**_BASE, height=180, showlegend=False,
                      xaxis=dict(title="seconds", **_GRID), yaxis=dict(range=[0, 1], title="model score", **_GRID))
    return fig
