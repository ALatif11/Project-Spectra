"""Mechanics regressions; generated waveforms are not speech accuracy evidence."""

from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from scripts.audit_inference import distribution, format_parity, read_cache
from src.audio import decode_audio
from src.dataset import load_splits
from src.inference import predict_audio
from src.model import load_model, segment_scores, window_features
from test_dataset import technical_dataset  # noqa: F401
from test_model import fitted  # noqa: F401


@pytest.mark.parametrize('rate', [16000, 48000])
@pytest.mark.parametrize('seconds', [0.25, 2.5, 7.339])
def test_lossless_container_and_resampling_parity(fitted, monkeypatch, rate, seconds):
    _, _, checkpoint, _, _ = fitted
    monkeypatch.setenv('ECHOPROOF_MODEL_PATH', str(checkpoint))
    time = np.arange(round(rate * seconds)) / rate
    signal = 0.2 * np.sin(2 * np.pi * 440 * time)
    # Quantize once: separate float encoders can choose different PCM rounding.
    pcm = np.round(np.column_stack((signal, signal * 0.5)) * 32768).astype(np.int16)
    payloads = []
    for fmt in ('WAV', 'FLAC'):
        buffer = BytesIO()
        sf.write(buffer, pcm, rate,
                 format=fmt, subtype='PCM_16')
        payloads.append(buffer.getvalue())
    a, b = [decode_audio(payload) for payload in payloads]
    np.testing.assert_array_equal(a.samples, b.samples)
    xa, wa, _ = window_features(a)
    xb, wb, _ = window_features(b)
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(wa, wb)
    ra, rb = [predict_audio(payload) for payload in payloads]
    assert ra == rb
    assert ra['status'] == 'ok'
    pipeline = load_model(checkpoint)['pipeline']
    index = np.flatnonzero(pipeline.classes_ == 1).item()
    np.testing.assert_array_equal(segment_scores(pipeline, xa), pipeline.predict_proba(xa)[:, index])
    assert ra['synthetic_score'] == pytest.approx(np.average(segment_scores(pipeline, xa), weights=wa))
    decoded, features = format_parity(payloads[0])
    np.testing.assert_array_equal(decoded['WAV'].samples, a.samples)
    np.testing.assert_array_equal(features['FLAC'], xa)


def test_training_cache_matches_inference_without_writing(fitted):
    root, splits, _, _, cache = fitted
    row = next(r for r in load_splits(splits) if r['split'] == 'train')
    before = {p.name: p.read_bytes() for p in cache.glob('*.npz')}
    old, weights = read_cache(root, row, cache)
    new, new_weights, _ = window_features(decode_audio((root / row['path']).read_bytes()))
    np.testing.assert_allclose(old, new, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(weights, new_weights)
    assert before == {p.name: p.read_bytes() for p in cache.glob('*.npz')}
    (root / row['path']).write_bytes((root / row['path']).read_bytes() + b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        read_cache(root, row, cache)


def test_diagnostic_distances_do_not_adjust_predictions(fitted):
    root, splits, checkpoint, _, cache = fitted
    row = next(r for r in load_splits(splits) if r['split'] == 'train')
    reference, _ = read_cache(root, row, cache)
    pipeline = load_model(checkpoint)['pipeline']
    features = reference.copy() + 1
    before = pipeline.predict_proba(features).copy()
    result = distribution(features, reference, pipeline)
    assert len(result['outside_real_training_range_per_window']) == len(features)
    assert np.isfinite([item['z'] for item in result['largest_standardized_departures']]).all()
    np.testing.assert_array_equal(pipeline.predict_proba(features), before)


def test_internal_silence_is_currently_scored_and_duration_weighted(fitted, monkeypatch):
    _, _, checkpoint, _, _ = fitted
    monkeypatch.setenv('ECHOPROOF_MODEL_PATH', str(checkpoint))
    time = np.arange(64000) / 16000
    samples = np.concatenate((0.2*np.sin(2*np.pi*440*time), np.zeros(64000)))
    buffer = BytesIO()
    sf.write(buffer, samples, 16000, format='WAV', subtype='PCM_16')
    result = predict_audio(buffer.getvalue())
    assert result['status'] == 'ok'
    assert len(result['segments']) == 2
    assert result['synthetic_score'] == pytest.approx(np.mean([s['synthetic_score'] for s in result['segments']]))
