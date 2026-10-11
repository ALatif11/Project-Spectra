import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import inthewild as itw
from src.model import FEATURE_SETTINGS, load_model, window_features
from src.audio import decode_audio


def catalogue():
    return [dict(path=f'{s}-{label}-{i}.wav',speaker=f'speaker-{s}',label=label,
                 shard='data/test-00000-of-00008.parquet',row=s*40+label*20+i,
                 utterance_id=f'{s}-{label}-{i}')
            for s in range(20) for label in (0,1) for i in range(20)]


def test_speaker_assignment_balanced_selection_reproducible():
    rows=itw.assign_speakers(catalogue())
    assert rows==itw.assign_speakers(catalogue())
    selected=itw.select_rows(rows,600)
    assert len(selected)==600
    assert all({r['label'] for r in selected if r['split']==s}=={0,1} for s in ['train','validation','test'])
    for speaker in {r['speaker'] for r in selected}:
        assert len({r['split'] for r in selected if r['speaker']==speaker})==1
    assert all(abs(sum(r['label']==0 and r['split']==s for r in selected)-sum(r['label']==1 and r['split']==s for r in selected))<=1 for s in ['train','validation','test'])


def test_missing_independent_speakers_rejected():
    with pytest.raises(ValueError):itw.assign_speakers([dict(speaker='one',label=i) for i in (0,1)])


def test_metadata_actual_label_and_speaker_checks():
    import pyarrow as pa
    rows=itw.metadata_rows(pa.Table.from_pylist([dict(path='4.wav',label=0,
          notes=json.dumps(dict(speaker='name',utterance_id='ITW_4',label='bona-fide')))]),'shard')
    assert rows[0]['label']==0
    with pytest.raises(ValueError):itw.metadata_rows(pa.Table.from_pylist([dict(path='4.wav',label=1,
          notes=json.dumps(dict(speaker='name',utterance_id='ITW_4',label='bona-fide')))]),'shard')


def test_optimization_refuses_test_and_crossing_speakers(tmp_path):
    rows=[dict(path='a',split='train',label=0,speaker='a'),dict(path='b',split='validation',label=1,speaker='a')]
    itw.save_json(tmp_path/'optimization.json',rows)
    with pytest.raises(ValueError,match='Speaker leakage'):itw.optimization_rows(tmp_path)
    (tmp_path/'optimization.json').unlink()
    itw.save_json(tmp_path/'optimization.json',[dict(path='test',split='test',label=0,speaker='a')])
    with pytest.raises(ValueError,match='test'):itw.optimization_rows(tmp_path)


def test_metrics_single_class_rates():
    r=itw.summary([0,0],[.1,.9],.5)
    assert r['false_positive_rate']==.5
    assert r['false_negative_rate'] is None
    assert r['balanced_accuracy'] is None


def test_range_refuses_full_response_and_large_read(monkeypatch):
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*a):pass
    import requests
    monkeypatch.setattr(requests,'get',lambda *a,**k:Response())
    r=itw.RangeFile('https://example.com',3_000_000)
    with pytest.raises(ValueError,match='budget'):r.read(2_000_001)
    with pytest.raises(ValueError,match='refused'):r.read(100)


def test_extraction_reuses_real_features_and_never_reads_sealed_test(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_limits
    root=tmp_path/'root';(root/'data').mkdir(parents=True);exp=tmp_path/'exp';exp.mkdir()
    (exp/'sealed').mkdir();(exp/'sealed'/'test.json').write_text('POISON TEST CONTENT')
    samples=(.1*np.sin(np.arange(8000)*.08)).astype(np.float32)
    b=io.BytesIO();sf.write(b,samples,16000,format='FLAC');payload=b.getvalue()
    sha=hashlib.sha256(payload).hexdigest();manifest=[];audio=[]
    for i,(split,label) in enumerate([('train',0),('train',1),('validation',0),('validation',1)]):
        manifest.append(dict(path=f'{i}.wav',split=split,label=label,speaker=split,
                             shard='data/test-00000-of-00008.parquet',sha256=sha))
        audio.append(dict(path=f'{i}.wav',audio=dict(bytes=payload,path=f'{i}.wav')))
    itw.save_json(exp/'optimization.json',manifest)
    pq.write_table(pa.Table.from_pylist(audio),root/'data/test-00000-of-00008.parquet')
    with threadpool_limits(limits=1):
        first=itw.extract(root,exp);second=itw.extract(root,exp)
        f,w=itw.read_features(exp,manifest[0]);expected,ew,_=window_features(decode_audio(payload))
    assert first['clips']==4 and first['cache_hits']==3
    assert second['cache_hits']==4
    np.testing.assert_array_equal(f,expected);np.testing.assert_array_equal(w,ew)


def test_compare_fit_training_only_and_preserve_baseline(tmp_path):
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(4);exp=tmp_path/'exp';exp.mkdir();(exp/'features').mkdir();rows=[]
    for i in range(60):
        label=i%2;split='train' if i<40 else 'validation'
        f=rng.normal(size=(1,52)).astype('float32')+label+(50 if split=='validation' else 0)
        sha=hashlib.sha256(str(i).encode()).hexdigest();key=hashlib.sha256((sha+json.dumps(FEATURE_SETTINGS,sort_keys=True)).encode()).hexdigest()
        np.savez(exp/'features'/f'{key}.npz',features=f,weights=[64000.])
        rows.append(dict(path=f'{i}.wav',label=label,split=split,speaker=f'{split}-{i//4}',sha256=sha))
    itw.save_json(exp/'optimization.json',rows)
    (exp/'sealed').mkdir();(exp/'sealed'/'test.json').write_text('NEVER PARSE')
    pipe=Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression())]);pipe.fit(rng.normal(size=(10,52)),np.arange(10)%2)
    baseline=tmp_path/'baseline.joblib';joblib.dump(dict(pipeline=pipe,threshold=.5,model_version='baseline-v1',feature_settings=FEATURE_SETTINGS),baseline)
    digest=hashlib.sha256(baseline.read_bytes()).hexdigest();model=tmp_path/'candidate.joblib'
    result=itw.compare(exp,baseline,model)
    art=load_model(model)
    x=np.concatenate([itw.read_features(exp,r)[0] for r in rows[:40]])
    np.testing.assert_allclose(art['pipeline'].named_steps['scaler'].mean_,x.mean(axis=0),rtol=1e-5,atol=1e-6)
    assert result['test_accessed'] is False and result['luminal_calls']==0
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==digest
    with pytest.raises(ValueError,match='Preserve'):itw.compare(exp,baseline,model)


def test_prepare_excludes_malformed_and_cross_speaker_duplicates(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=tmp_path/'root';(root/'data').mkdir(parents=True);cat=[];audio=[];rng=np.random.default_rng(5)
    payloads=[]
    for i in range(48):
        b=io.BytesIO();sf.write(b,rng.normal(0,.03,5000),16000,format='FLAC');payloads.append(b.getvalue())
    payloads[4]=payloads[0]  # Two speakers, same genuine waveform: conservatively exclude both speakers.
    payloads[47]=b'bad file'
    for i in range(48):
        path=f'{i}.wav';label=i%2;speaker=f'speaker-{i//4}'
        cat.append(dict(path=path,label=label,speaker=speaker,utterance_id=f'ITW_{i}',row=i,shard='data/test-00000-of-00008.parquet'))
        audio.append(dict(path=path,label=label,audio=dict(bytes=payloads[i],path=path),
            notes=json.dumps(dict(speaker=speaker,utterance_id=f'ITW_{i}',label='spoof' if label else 'bona-fide'))))
    schema=pa.Table.from_pylist(audio).schema.with_metadata({b'huggingface':json.dumps(dict(info=dict(features=dict(
        path={},audio=dict(sampling_rate=16000),label=dict(names=['bonafide','spoof']),notes={})))).encode()})
    pq.write_table(pa.Table.from_pylist(audio,schema=schema),root/'data/test-00000-of-00008.parquet')
    itw.save_json(root/'catalogue.json',cat);exp=tmp_path/'exp'
    result=itw.prepare(root,exp,target=300)
    assert result['invalid_excluded'][0]['path']=='47.wav'
    assert result['duplicates_excluded']==1
    assert result['cross_speaker_duplicate_speakers_excluded']==['speaker-0','speaker-1']
    assert result['selected']['n']==39
    trainval=itw.optimization_rows(exp);test=json.loads((exp/'sealed/test.json').read_text())
    assert not {r['speaker'] for r in trainval}&{r['speaker'] for r in test}
