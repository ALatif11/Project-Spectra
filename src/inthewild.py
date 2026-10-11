"""Bounded, opt-in InTheWild experiment; never used by app startup.

Only speaker identity is published: recording/session/vendor grouping is unknown.
The upstream split is named test; local train/validation/sealed-test memberships
are new, and results are not an untouched upstream benchmark result.
"""
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

REPO = 'SpeechAntiSpoofingBenchmarks/InTheWild'
REVISION = 'f36fbe7e4e009e444a4985c66d76ab1d9439f901'
LABELS = {'bonafide': 0, 'spoof': 1}
LIMITATIONS = ('Speaker-disjoint local partitions; source recording/session, TTS vendor, '
               'shared text and near-duplicate provenance are unavailable. Cross-speaker '
               'source leakage cannot be ruled out. Public-figure English web audio is '
               'not representative of arbitrary microphones/languages. Uncalibrated '
               'scores are not forensic proof. Upstream test is repurposed for this '
               'experiment and is no longer an independent benchmark.')


def save_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


class RangeFile(io.RawIOBase):
    """Seekable remote reader refusing full responses and >10 MB/shard metadata."""
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.total = url, size, 0, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('Negative seek')
        return self.pos
    def read(self, n=-1):
        import requests
        n = min(n if n >= 0 else self.size - self.pos, self.size - self.pos)
        if n <= 0: return b''
        if n > 2_000_000 or self.total + n > 10_000_000:
            raise ValueError('Metadata range budget exceeded')
        end = self.pos + n - 1
        with requests.get(self.url + f'?range={self.pos}-{end}',
                          headers={'Range': f'bytes={self.pos}-{end}'},
                          stream=True, timeout=(10, 30)) as r:
            if r.status_code != 206: raise ValueError(f'Range request refused: HTTP {r.status_code}')
            b = r.raw.read(n + 1)
        if len(b) != n: raise ValueError('Incorrect range response length')
        self.total += n
        self.pos += n
        return b


def shard_url(name):
    return f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}'


def verify_schema(schema):
    features = json.loads(schema.metadata[b'huggingface'])['info']['features']
    if (set(schema.names) != {'path', 'audio', 'label', 'notes'}
            or features['label']['names'] != ['bonafide', 'spoof']
            or features['audio']['sampling_rate'] != 16000):
        raise ValueError('Unexpected published schema/label mapping')
    return features


def metadata_rows(table, shard):
    rows = []
    for i, item in enumerate(table.to_pylist()):
        notes = json.loads(item['notes'])
        raw_label = notes['label']
        expected = {'bona-fide': 0, 'spoof': 1}.get(raw_label)
        if expected != item['label'] or not notes.get('speaker') or not notes.get('utterance_id'):
            raise ValueError('Metadata label/speaker mismatch')
        rows.append(dict(path=item['path'], label=expected, speaker=notes['speaker'].strip(),
                         utterance_id=notes['utterance_id'], shard=shard, row=i))
    return rows


def distribution(rows):
    speakers = defaultdict(Counter)
    for r in rows: speakers[r['speaker']][str(r['label'])] += 1
    return dict(n=len(rows), classes=dict(Counter(str(r['label']) for r in rows)),
                speakers=len(speakers), per_speaker={k: dict(v) for k,v in sorted(speakers.items())})


def inspect(root):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    root.mkdir(parents=True, exist_ok=True)
    if (root/'catalogue.json').exists(): return json.loads((root/'inspection.json').read_text())
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    shards = [dict(name=s.rfilename, bytes=s.size) for s in info.siblings
              if s.rfilename.startswith('data/test-') and s.rfilename.endswith('.parquet')]
    rows, byte_count, schemas = [], 0, []
    for shard in sorted(shards, key=lambda s:s['name']):
        with RangeFile(shard_url(shard['name']), shard['bytes']) as remote:
            pf = pq.ParquetFile(remote)
            schemas.append(verify_schema(pf.schema_arrow))
            current = metadata_rows(pf.read(columns=['path','label','notes']), shard['name'])
            rows.extend(current); byte_count += remote.total
        print(f"Metadata {shard['name']}: {len(current)} rows", flush=True)
    if len({r['path'] for r in rows}) != len(rows) or len({r['utterance_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate published identities')
    report = dict(repo=REPO, revision=REVISION, schema=schemas[0], label_mapping=LABELS,
                  shards=shards, metadata_transfer_bytes=byte_count,
                  distribution=distribution(rows), limitations=LIMITATIONS)
    save_json(root/'catalogue.json', rows); save_json(root/'inspection.json', report)
    return report


def download(root, count, max_seconds=300):
    """Download at most four selected shards; stream to disk, no unrelated files."""
    import requests
    import shutil
    info = json.loads((root/'inspection.json').read_text())
    if not 1 <= count <= 4: raise ValueError('Download capped at four shards')
    shards = sorted(info['shards'], key=lambda s:s['name'])[:count]
    needed = sum(s['bytes'] for s in shards if not (root/s['name']).exists())
    if shutil.disk_usage(root).free < needed + 2_000_000_000:
        raise ValueError('Insufficient disk headroom')
    started = time.monotonic()
    for shard in shards:
        path = root/shard['name']; path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size != shard['bytes']: raise ValueError('Existing shard size mismatch')
            continue
        temporary = path.with_suffix('.partial')
        with requests.get(shard_url(shard['name']), stream=True, timeout=(10,30)) as r:
            r.raise_for_status()
            size = 0
            with temporary.open('wb') as f:
                for chunk in r.iter_content(1024*1024):
                    size += len(chunk)
                    if size > shard['bytes'] or time.monotonic()-started > max_seconds:
                        raise ValueError('Download size/runtime limit reached; partial file retained')
                    f.write(chunk)
                    if size // (32*1024*1024) != (size-len(chunk)) // (32*1024*1024):
                        print(f"Download {path.name}: {size/1e6:.0f}/{shard['bytes']/1e6:.0f} MB",flush=True)
        if size != shard['bytes']: raise ValueError('Incomplete shard download')
        os.link(temporary, path); temporary.unlink()
    return dict(shards=count, bytes=sum(s['bytes'] for s in shards), seconds=time.monotonic()-started)


def assign_speakers(rows, seed=42):
    """Choose partitions using counts only, before any features or predictions."""
    speakers = sorted({r['speaker'] for r in rows})
    counts = np.array([[sum(r['speaker']==s and r['label']==label for r in rows)
                        for label in (0,1)] for s in speakers])
    rng = np.random.default_rng(seed); target = np.array([.7,.15,.15])
    best, best_loss = None, float('inf')
    for _ in range(4096):
        choice = rng.choice(3,len(speakers),p=target)
        totals = np.array([counts[choice==i].sum(axis=0) for i in range(3)])
        if (totals == 0).any(): continue
        loss = np.square(totals/counts.sum(axis=0)-target[:,None]).sum()
        if loss < best_loss: best,best_loss = choice.copy(), loss
    if best is None: raise ValueError('Need enough independent speakers in both classes')
    mapping = dict(zip(speakers, best))
    return [dict(r, split=('train','validation','test')[mapping[r['speaker']]],seed=seed) for r in rows]


def select_rows(rows, target, seed=42):
    """Near balance within each partition; round-robin speakers for coverage."""
    rng = np.random.default_rng(seed); selected=[]
    for split, ratio in [('train',.7),('validation',.15),('test',.15)]:
        pools = {label:[r for r in rows if r['split']==split and r['label']==label] for label in (0,1)}
        budget = min(round(target*ratio), sum(map(len,pools.values())))
        half = budget//2
        quotas = {label:min(half,len(pools[label])) for label in (0,1)}
        for label in (0,1): quotas[label] += min(budget-sum(quotas.values()),len(pools[label])-quotas[label])
        for label in (0,1):
            buckets=defaultdict(list)
            for r in pools[label]: buckets[r['speaker']].append(r)
            for bucket in buckets.values(): rng.shuffle(bucket)
            names=sorted(buckets); rng.shuffle(names); chosen=[]
            while len(chosen)<quotas[label]:
                for s in names:
                    if buckets[s] and len(chosen)<quotas[label]: chosen.append(buckets[s].pop())
            selected.extend(chosen)
    return sorted(selected,key=lambda r:(r['shard'],r['row']))


def prepare(root, experiment, target=12000, seed=42, max_seconds=300):
    import pyarrow.parquet as pq
    import soundfile as sf
    from src.audio import decode_audio
    started = time.monotonic()
    if experiment.exists(): raise ValueError('Preserve existing experiment; use a new path')
    rows = json.loads((root/'catalogue.json').read_text())
    rows = [r for r in rows if (root/r['shard']).exists()]
    if not rows: raise ValueError('Explicitly download selected shards first')
    if not 300 <= target <= 12000: raise ValueError('Subset size must be 300..12000')
    # Decode one bounded batch at a time for PCM identity, not model evaluation.
    hashes, seen, duplicates, invalid = {}, {}, [], []
    expected_labels={r['path']:r['label'] for r in rows}
    for shard in sorted({r['shard'] for r in rows}):
        pf=pq.ParquetFile(root/shard); verify_schema(pf.schema_arrow)
        offset=0
        for batch in pf.iter_batches(batch_size=16, columns=['audio','path','label']):
            for item in batch.to_pylist():
                offset+=1
                if time.monotonic()-started > max_seconds:
                    raise ValueError('Identity audit runtime budget reached')
                if expected_labels.get(item['path']) != item['label']:
                    raise ValueError('Downloaded labels differ from inspected metadata')
                try:
                    info=sf.info(io.BytesIO(item['audio']['bytes']))
                    if info.samplerate != 16000 or info.channels != 1:
                        raise ValueError('Published audio is not 16 kHz mono')
                    audio=item['audio']['bytes']
                    rec=decode_audio(audio)
                    if rec.is_silent or rec.duration_s<.25:
                        raise ValueError('Silent or shorter than 0.25 seconds')
                except (ValueError, RuntimeError) as exc:
                    invalid.append(dict(path=item['path'],reason=str(exc)))
                    continue
                pcm=hashlib.sha256(rec.samples.tobytes()).hexdigest()
                previous=seen.get(pcm)
                if previous:
                    if previous[1]!=item['label']: raise ValueError('Identical PCM has conflicting labels')
                    duplicates.append(item['path'])
                    # Merge speakers linked by duplicates before assigning splits below.
                else: seen[pcm]=(item['path'],item['label'])
                hashes[item['path']]=dict(sha256=hashlib.sha256(audio).hexdigest(),pcm_sha256=pcm)
        print(f'Identity audit {shard}: {offset} clips',flush=True)
    # Conservatively drop every speaker connected by a cross-speaker PCM duplicate.
    owners=defaultdict(set)
    for r in rows:
        if r['path'] in hashes: owners[hashes[r['path']]['pcm_sha256']].add(r['speaker'])
    ambiguous={s for names in owners.values() if len(names)>1 for s in names}
    retained=[r for r in rows if r['path'] in hashes and r['path'] not in duplicates and r['speaker'] not in ambiguous]
    selected=select_rows(assign_speakers(retained,seed),target,seed)
    experiment.mkdir(parents=True,exist_ok=False)
    for r in selected: r.update(hashes[r['path']])
    save_json(experiment/'optimization.json',[r for r in selected if r['split']!='test'])
    sealed=experiment/'sealed';sealed.mkdir();save_json(sealed/'test.json',[r for r in selected if r['split']=='test'])
    report=dict(repo=REPO,revision=REVISION,label_mapping=LABELS,available=distribution(rows),
                identity_audit_seconds=time.monotonic()-started,
                downloaded_audio_bytes=sum((root/s).stat().st_size for s in {r['shard'] for r in rows}),
                selected=distribution(selected),duplicates_excluded=len(duplicates),invalid_excluded=invalid,
                cross_speaker_duplicate_speakers_excluded=sorted(ambiguous),seed=seed,
                partitions={p:distribution([r for r in selected if r['split']==p]) for p in ('train','validation','test')},
                limitations=LIMITATIONS)
    save_json(experiment/'preparation.json',report)
    return report


def optimization_rows(experiment):
    rows=json.loads((experiment/'optimization.json').read_text())
    memberships={}
    for r in rows:
        if r['split'] not in ('train','validation') or r['label'] not in (0,1):
            raise ValueError('Optimization manifest contains test or invalid labels')
        if memberships.setdefault(r['speaker'],r['split']) != r['split']:
            raise ValueError('Speaker leakage')
    for split in ('train','validation'):
        if {r['label'] for r in rows if r['split']==split}!={0,1}: raise ValueError('Both classes required')
    return rows


def extract(root, experiment, limit=None, max_seconds=600):
    import pyarrow.parquet as pq
    import resource
    import sys
    from src.audio import decode_audio
    from src.model import FEATURE_SETTINGS, window_features
    started=time.monotonic(); rows=optimization_rows(experiment)
    if limit:
        if not 1 <= limit <= 500: raise ValueError('Pilot feature count must be 1..500')
        # Benchmark coverage of train/validation and both classes, not shard prefix alone.
        pools=[[r for r in rows if r['split']==s and r['label']==l] for s in ('train','validation') for l in (0,1)]
        rows=[r for pool in pools for r in pool[:max(1,limit//4)]]
    wanted={r['path']:r for r in rows};cache=experiment/'features';cache.mkdir(exist_ok=True)
    policy=json.dumps(FEATURE_SETTINGS,sort_keys=True)
    done=hits=windows=0
    for shard in sorted({r['shard'] for r in rows}):
        pf=pq.ParquetFile(root/shard)
        for batch in pf.iter_batches(batch_size=16,columns=['path','audio']):
            for item in batch.to_pylist():
                r=wanted.get(item['path'])
                if r is None: continue
                if time.monotonic()-started>max_seconds: raise ValueError('Feature runtime budget reached; resume cached work')
                key=hashlib.sha256((r['sha256']+policy).encode()).hexdigest(); path=cache/f'{key}.npz'
                if path.exists(): hits+=1
                else:
                    payload=item['audio']['bytes']
                    if hashlib.sha256(payload).hexdigest()!=r['sha256']:raise ValueError('Audio changed')
                    f,w,_=window_features(decode_audio(payload));temp=path.with_suffix('.tmp.npz')
                    np.savez_compressed(temp,features=f,weights=w);os.link(temp,path);temp.unlink()
                with np.load(path,allow_pickle=False) as c: windows+=len(c['weights'])
                done+=1
                if done%100==0: print(f'Features {done}/{len(rows)} clips ({time.monotonic()-started:.1f}s, {hits} cached)',flush=True)
    if done!=len(rows):raise ValueError('Some selected audio missing')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=peak if sys.platform=='darwin' else peak*1024
    return dict(clips=done,windows=windows,cache_hits=hits,seconds=time.monotonic()-started,
                peak_process_rss_mb=peak_bytes/1e6,
                clips_per_second=done/(time.monotonic()-started),feature_settings=FEATURE_SETTINGS)


def read_features(experiment,row):
    from src.model import FEATURE_SETTINGS
    policy=json.dumps(FEATURE_SETTINGS,sort_keys=True)
    key=hashlib.sha256((row['sha256']+policy).encode()).hexdigest()
    with np.load(experiment/'features'/f'{key}.npz',allow_pickle=False) as c:
        f,w=c['features'],c['weights']
    if f.ndim!=2 or f.shape[1]!=52 or w.shape!=(len(f),) or not np.isfinite(f).all() or not (w>0).all():
        raise ValueError('Invalid cached features')
    return f,w


def summary(labels,scores,threshold):
    from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score
    labels=np.asarray(labels); pred=np.asarray(scores)>=threshold
    tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return dict(n=len(labels),threshold=float(threshold),
                balanced_accuracy=float(balanced_accuracy_score(labels,pred)) if len(set(labels))==2 else None,
                f1_synthetic=float(f1_score(labels,pred,zero_division=0)),
                false_positive_rate=float(fp/(tn+fp)) if tn+fp else None,
                false_negative_rate=float(fn/(fn+tp)) if fn+tp else None,
                confusion_matrix_real_fake=[[int(tn),int(fp)],[int(fn),int(tp)]])


def compare(experiment, baseline_path, model_path, diagnostics=(), max_seconds=120):
    import joblib, sklearn, warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from src.audio import decode_audio
    from src.model import FEATURE_SETTINGS,file_digest,load_model,segment_scores,window_features
    from src.training import select_threshold
    if model_path.exists() or (experiment/'comparison.json').exists():raise ValueError('Preserve prior comparison/model')
    rows=optimization_rows(experiment); train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='validation']
    baseline_hash=file_digest(baseline_path); baseline=load_model(baseline_path)
    arrays=[]; labels=[]; weights=[]
    for r in train:
        f,w=read_features(experiment,r); arrays.append(f); labels.extend([r['label']]*len(f)); weights.extend(w/w.sum())
    pipe=Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression(C=1,max_iter=1000,solver='lbfgs',random_state=42))])
    started=time.monotonic()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipe.fit(np.concatenate(arrays),np.array(labels),scaler__sample_weight=np.array(weights),classifier__sample_weight=np.array(weights))
    fit_seconds=time.monotonic()-started
    if fit_seconds>max_seconds:raise ValueError('Fit exceeded runtime budget; candidate not exported')
    def score(p):
        return np.array([float(np.average(segment_scores(p,f),weights=w)) for f,w in (read_features(experiment,r) for r in val)])
    old,new=score(baseline['pipeline']),score(pipe); y=np.array([r['label'] for r in val]);threshold=select_threshold(y,new)
    report=dict(partition='validation',baseline_model=str(baseline_path),baseline_sha256=baseline_hash,
                baseline=summary(y,old,baseline['threshold']),candidate=summary(y,new,threshold),
                candidate_at_05=summary(y,new,.5),training_clips=len(train),training_windows=len(labels),
                training_seconds=fit_seconds,original_training_included=False,limitations=LIMITATIONS,
                per_speaker={},external_diagnostics=[],test_accessed=False,luminal_calls=0,
                deployed=False,candidate_model=str(model_path),
                threshold_selection='Maximum validation balanced accuracy; ties nearest 0.5')
    for speaker in sorted({r['speaker'] for r in val}):
        mask=np.array([r['speaker']==speaker for r in val]);report['per_speaker'][speaker]=dict(baseline=summary(y[mask],old[mask],baseline['threshold']),candidate=summary(y[mask],new[mask],threshold))
    for path,truth in diagnostics:
        f,w,_=window_features(decode_audio(path.read_bytes())); a=float(np.average(segment_scores(baseline['pipeline'],f),weights=w));b=float(np.average(segment_scores(pipe,f),weights=w))
        report['external_diagnostics'].append(dict(path=str(path),label=truth,label_source='previous audit/user attestation',sha256=file_digest(path),baseline_score=a,candidate_score=b,baseline_error=bool((a>=baseline['threshold'])!=truth),candidate_error=bool((b>=threshold)!=truth),used_for_selection=False))
    if file_digest(baseline_path)!=baseline_hash:raise ValueError('Baseline changed during experiment')
    artifact=dict(pipeline=pipe,threshold=threshold,model_version='inthewild-v1',feature_settings=FEATURE_SETTINGS,
                  label_mapping=LABELS,seed=42,sklearn_version=sklearn.__version__,validation=report,
                  train_groups=sorted({r['speaker'] for r in train}),validation_groups=sorted({r['speaker'] for r in val}),
                  splits_sha256=file_digest(experiment/'optimization.json'),
                  provenance=dict(repo=REPO,revision=REVISION,baseline_sha256=baseline_hash),
                  inference_metadata=dict(score='uncalibrated synthetic class output',abstention_band=.05))
    model_path.parent.mkdir(parents=True,exist_ok=True);temp=model_path.with_suffix('.tmp.joblib')
    try:
        joblib.dump(artifact,temp);load_model(temp);os.link(temp,model_path)
    finally:temp.unlink(missing_ok=True)
    save_json(experiment/'comparison.json',report)
    return report
