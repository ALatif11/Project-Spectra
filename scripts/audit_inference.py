"""Read-only inference diagnosis. Never fits, downloads, or changes old artifacts."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.audio import decode_audio
from src.dataset import load_splits
from src.features import FEATURE_NAMES
from src.inference import predict_audio
from src.model import FEATURE_SETTINGS, MODEL_PATH, file_digest, load_model, segment_scores, window_features


def read_cache(root, row, cache):
    """Require existing training caches; don't regenerate historical features."""
    if file_digest(root / row['path']) != row['sha256']:
        raise ValueError(f"Dataset bytes changed: {row['path']}")
    policy = json.dumps(FEATURE_SETTINGS, sort_keys=True).encode()
    key = hashlib.sha256(row['sha256'].encode() + policy).hexdigest()
    with np.load(cache / f'{key}.npz', allow_pickle=False) as data:
        return data['features'].copy(), data['weights'].copy()


def distribution(features, reference, pipeline):
    """Descriptive distances, not an OOD gate or calibrated certainty."""
    reference = reference.astype(np.float64)
    mean, std = reference.mean(axis=0), reference.std(axis=0)
    low, high = reference.min(axis=0), reference.max(axis=0)
    nonconstant = std > 0
    z = np.zeros_like(features, dtype=float)
    z[:, nonconstant] = (features[:, nonconstant] - mean[nonconstant]) / std[nonconstant]
    outside = (features < low) | (features > high)
    contribution = pipeline.named_steps['scaler'].transform(features) * pipeline.named_steps['classifier'].coef_[0]
    top = np.argsort(np.max(np.abs(z), axis=0))[-10:][::-1]
    positive = np.argsort(contribution.mean(axis=0))[-10:][::-1]
    return {
        'reference': 'Real training windows only; unweighted descriptive moments',
        'outside_real_training_range_per_window': outside.sum(axis=1).tolist(),
        'largest_standardized_departures': [dict(feature=FEATURE_NAMES[i],
            values=features[:, i].tolist(), real_mean=float(mean[i]), real_std=float(std[i]),
            real_min=float(low[i]), real_max=float(high[i]), z=z[:, i].tolist()) for i in top],
        'largest_positive_logit_terms': [dict(feature=FEATURE_NAMES[i],
            contribution=contribution[:, i].tolist()) for i in positive],
        'window_logits': pipeline.decision_function(features).tolist(),
        'intercept': float(pipeline.named_steps['classifier'].intercept_[0]),
        'warning': 'Correlated feature contributions are descriptive, not causal evidence of synthesis.',
    }


def format_parity(payload):
    """Compare the same PCM16 samples in two lossless containers."""
    samples, rate = sf.read(BytesIO(payload), dtype='int16', always_2d=True)
    decoded, features = {}, {}
    for fmt in ('WAV', 'FLAC'):
        buffer = BytesIO()
        sf.write(buffer, samples, rate, format=fmt, subtype='PCM_16')
        decoded[fmt] = decode_audio(buffer.getvalue())
        features[fmt] = window_features(decoded[fmt])[0]
    return decoded, features


def audit(root, splits, cache, model_path, external):
    artifact = load_model(model_path)
    pipeline = artifact['pipeline']
    if artifact['splits_sha256'] != file_digest(splits):
        raise ValueError('Manifest differs from the fitted checkpoint.')
    rows = load_splits(splits)
    with splits.open(newline='') as stream:
        raw = list(csv.DictReader(stream))
    train = [row for row in rows if row['split'] == 'train']
    arrays, weights, real = [], [], []
    for row in train:
        values, duration = read_cache(root, row, cache)
        arrays.append(values)
        weights.extend(duration / duration.sum())
        if int(row['label']) == 0:
            real.append(values)
    x, w = np.concatenate(arrays), np.asarray(weights)
    mean = np.average(x, axis=0, weights=w)
    variance = np.average((x - mean) ** 2, axis=0, weights=w)
    scaler = pipeline.named_steps['scaler']
    parity = []
    # Only train/validation diagnostics; don't rescore the final test.
    for partition in ('train', 'validation'):
        for label in ('0', '1'):
            row = next(r for r in rows if r['split'] == partition and r['label'] == label)
            old, duration = read_cache(root, row, cache)
            fresh, fresh_duration, _ = window_features(decode_audio((root / row['path']).read_bytes()))
            parity.append(dict(path=row['path'], split=partition,
                max_feature_difference=float(np.max(np.abs(old - fresh))),
                identical_weights=bool(np.array_equal(duration, fresh_duration))))
    memberships = defaultdict(set)
    for row in raw:
        memberships[row['leakage_group']].add(row['split'])
    report = {
        'model': dict(path=str(model_path.resolve()), sha256=file_digest(model_path),
            version=artifact['model_version'], classes=pipeline.classes_.tolist(),
            threshold=artifact['threshold'], dimensions=pipeline.n_features_in_,
            classifier=pipeline.named_steps['classifier'].get_params(), feature_settings=FEATURE_SETTINGS),
        'scaler': dict(mean_max_abs_difference=float(np.max(np.abs(mean - scaler.mean_))),
            variance_max_abs_difference=float(np.max(np.abs(variance - scaler.var_))),
            variance_max_relative_difference=float(np.max(np.abs(variance - scaler.var_) / np.maximum(scaler.var_, 1e-20))),
            mean_matches=bool(np.allclose(mean, scaler.mean_, rtol=1e-6, atol=1e-6)),
            variance_matches=bool(np.allclose(variance, scaler.var_, rtol=1e-6, atol=1e-6)),
            train_clip_weight_sum=float(w.sum()), fitted_weight_sum=float(scaler.n_samples_seen_)),
        'cache_parity': parity,
        'split_audit': dict(sha256=file_digest(splits), seed=raw[0]['seed'],
            raw_clips=len(raw), retained_clips=len(rows), excluded_duplicates=len(raw)-len(rows),
            crossing_components=[g for g, parts in memberships.items() if len(parts)>1],
            partitions={p: dict(classes=dict(Counter(r['label'] for r in rows if r['split']==p)),
                real_sources=sorted({r['group'] for r in rows if r['split']==p and r['label']=='0'}),
                vendors=dict(Counter(r['vendor'] for r in rows if r['split']==p)))
                for p in ('train', 'validation', 'test')},
            excluded_by_vendor=dict(Counter(r['vendor'] for r in raw if r['duplicate_of'])),
            original_rates_channels=dict(Counter(f"{r['source_sample_rate']} Hz/{r['source_channels']} channels" for r in raw)),
            limitation='Filename source IDs and recorded PCM duplicates verified; shared speakers/text, near duplicates, and vendor provenance not independently verified.'),
        'external_diagnostic_set': [],
        'external_note': 'User-provided labels; diagnostic samples remain separate from fitting. No external synthetic recordings supplied unless listed.',
    }
    with patch.dict(os.environ, {'ECHOPROOF_MODEL_PATH': str(model_path.resolve())}):
        for path, label in external:
            payload = path.read_bytes()
            info = sf.info(BytesIO(payload))
            recording = decode_audio(payload)
            features, duration, _ = window_features(recording)
            decoded, encoded_features = format_parity(payload)
            recording_scores = {fmt: float(np.average(segment_scores(pipeline, f), weights=duration))
                                for fmt, f in encoded_features.items()}
            report['external_diagnostic_set'].append(dict(path=str(path.resolve()),
                ground_truth=label, label_source='user attestation', sha256=hashlib.sha256(payload).hexdigest(),
                original=dict(sample_rate=info.samplerate, channels=info.channels, subtype=info.subtype),
                duration_s=recording.duration_s, result=predict_audio(payload, path.name),
                format_parity=dict(comparison='Matched PCM16 WAV versus FLAC; lossy/float conversion may change samples',
                    waveform_max_abs_difference=float(np.max(np.abs(decoded['WAV'].samples-decoded['FLAC'].samples))),
                    feature_max_abs_difference=float(np.max(np.abs(encoded_features['WAV']-encoded_features['FLAC']))),
                    scores=recording_scores), distribution=distribution(features, np.concatenate(real), pipeline)))
    items = report['external_diagnostic_set']
    def counts(label):
        usable = [r for r in items if r['ground_truth']==label and r['result']['status']=='ok']
        return dict(supplied=sum(r['ground_truth']==label for r in items), scored=len(usable),
            threshold_errors=sum((r['result']['synthetic_score'] >= artifact['threshold']) != bool(label) for r in usable),
            ui_errors=sum(r['result']['label']==('likely_synthetic' if label==0 else 'likely_human') for r in usable),
            inconclusive=sum(r['result']['label']=='inconclusive' for r in usable))
    report['external_counts'] = dict(genuine=counts(0), synthetic=counts(1))
    return report


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--root', type=Path, default=Path('data/deepfake-audio-detection'))
    cli.add_argument('--splits', type=Path, default=Path('data/splits.csv'))
    cli.add_argument('--cache', type=Path, default=Path('data/features'))
    cli.add_argument('--model', type=Path, default=MODEL_PATH)
    cli.add_argument('--real', type=Path, action='append', default=[])
    cli.add_argument('--synthetic', type=Path, action='append', default=[])
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    # Exclusive creation prevents replacement of any existing audit/history.
    if args.output.exists():
        cli.error('Preserve existing reports; choose a new output path.')
    report = audit(args.root, args.splits, args.cache, args.model,
                   [(p, 0) for p in args.real] + [(p, 1) for p in args.synthetic])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Audit saved: {args.output}')


if __name__ == '__main__':
    main()
