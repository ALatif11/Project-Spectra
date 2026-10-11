"""Plotly figures computed locally from decoded audio (no model scores here)."""

import librosa
import numpy as np
import plotly.graph_objects as go

from src.audio import SAMPLE_RATE, AudioRecording, split_windows
from src.spectrogram import HOP_LENGTH, N_MELS, log_mel_spectrogram

MINT, VIOLET, MUTE = "#5eead4", "#a78bfa", "#8b8b95"
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8),
    font=dict(family="Geist, sans-serif", color=MUTE, size=11),
)
_GRID = dict(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.08)")


def waveform_fig(rec: AudioRecording, bins: int = 1600) -> go.Figure:
    x = rec.samples
    n = max(1, len(x) // bins)
    trim = x[: n * (len(x) // n)].reshape(-1, n)
    t = (np.arange(trim.shape[0]) * n + n / 2) / SAMPLE_RATE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=trim.max(1), line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=trim.min(1), fill="tonexty", fillcolor="rgba(94,234,212,.35)",
                             line=dict(color=MINT, width=1), hovertemplate="%{x:.2f}s<extra></extra>"))
    fig.update_layout(**_BASE, height=190, showlegend=False,
                      xaxis=dict(title="seconds", **_GRID), yaxis=dict(range=[-1, 1], **_GRID))
    return fig


def spectrogram_fig(rec: AudioRecording) -> go.Figure:
    """Log-Mel (dB, fixed unit-power reference) stitched across the 4 s windows, padding removed."""
    parts = []
    for w in split_windows(rec.samples):
        mel = log_mel_spectrogram(w.samples)
        parts.append(mel[:, : w.valid_samples // HOP_LENGTH + 1])
    mel = np.hstack(parts)
    t = np.arange(mel.shape[1]) * HOP_LENGTH / SAMPLE_RATE
    hz = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=SAMPLE_RATE / 2)[1:-1]
    fig = go.Figure(go.Heatmap(
        z=mel, x=t, y=hz, zmin=-80, zmax=float(max(mel.max(), -70)),
        colorscale=[[0, "#050505"], [.35, "#2e1b6b"], [.65, VIOLET], [.85, MINT], [1, "#f4fffc"]],
        colorbar=dict(title="dB", thickness=8, outlinewidth=0, tickfont=dict(size=10)),
        hovertemplate="%{x:.2f}s · %{y:.0f} Hz · %{z:.1f} dB<extra></extra>",
    ))
    fig.update_layout(**_BASE, height=300, xaxis=dict(title="seconds", **_GRID), yaxis=dict(title="Hz", **_GRID))
    return fig


def segments_fig(segments: list[dict], threshold: float | None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[(s["start_s"] + s["end_s"]) / 2 for s in segments],
        width=[s["end_s"] - s["start_s"] - 0.1 for s in segments],
        y=[s["synthetic_score"] for s in segments], marker=dict(color=VIOLET),
        hovertemplate="%{x:.1f}s · model score %{y:.3f}<extra></extra>",
    ))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=MINT, dash="dot", width=1),
                      annotation_text="threshold", annotation_font_color=MINT)
    fig.update_layout(**_BASE, height=170, showlegend=False,
                      xaxis=dict(title="seconds", **_GRID), yaxis=dict(range=[0, 1], title="model score", **_GRID))
    return fig
