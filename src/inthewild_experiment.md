# Bounded InTheWild experiment

This optional workflow uses the published
[SpeechAntiSpoofingBenchmarks/InTheWild repository](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/InTheWild)
at revision `f36fbe7e4e009e444a4985c66d76ab1d9439f901`. Its HF ClassLabel metadata
and every notes label are checked: bonafide=0, spoof=1. Speaker names are
available; recording/session IDs, TTS vendor and shared text are not.
Müller et al., *Does Audio Deepfake Detection Generalize?*, Interspeech 2022,
is the original dataset citation. The redistribution's Apache-2.0 license and
card are preserved under `data/inthewild/inspection/`.

Metadata inspection reads only path, label and notes with bounded range
requests across all eight shards. No audio is automatically downloaded on
import or app startup. Downloads are explicit, stream to disk, and are capped
at four shards (about 1.13 GB) and ten minutes. Partial files are never accepted
as complete. Preparation reads batches of 16 clips, verifies actual 16 kHz mono,
hashes normalized PCM, excludes duplicates, and conservatively excludes
speakers linked by cross-speaker duplicates before partitioning. Source overlap
between different speakers, near duplicates and cross-dataset overlap remain
unknown. No original Gary Stafford partition is included in candidate fitting.

Seed 42 selects whole speakers into approximately 70/15/15 partitions using
class counts only. Selection aims for balanced classes within each partition
and cycles speakers for coverage. Membership is frozen before any predictions.
Only train/validation rows appear in `optimization.json`; new test rows are
stored separately in `sealed/test.json`. Optimization and comparison do not
open that file or extract its features. Public upstream `test` is repurposed
for this experiment; resulting models cannot claim an untouched InTheWild
benchmark result. Keep the local test sealed through any future optimization.

Install optional dependencies only if missing:

```sh
.venv/bin/python -m pip install -r scripts/requirements-inthewild.txt
.venv/bin/python -m scripts.inthewild_experiment inspect
.venv/bin/python -m scripts.inthewild_experiment download --shards 1 --seconds 300
.venv/bin/python -m scripts.inthewild_experiment prepare --samples 600 --experiment data/inthewild/pilot-600-v1
.venv/bin/python -m scripts.inthewild_experiment extract --limit 300 --seconds 120 --experiment data/inthewild/pilot-600-v1
```

Inspect `benchmark.json` before proceeding. If the pilot is practical:

```sh
.venv/bin/python -m scripts.inthewild_experiment download --shards 4 --seconds 300
.venv/bin/python -m scripts.inthewild_experiment prepare --samples 12000
.venv/bin/python -m scripts.inthewild_experiment extract --seconds 600
.venv/bin/python -m scripts.inthewild_experiment compare --real /absolute/path/to/previously-audited-microphone.wav
```

All 52 existing features, local preprocessing, four-second windows, padding
and duration-weighted inference aggregation are reused exactly. Caches key
both audio bytes and feature policy. Scaler and logistic regression are fitted
on train windows only, with each clip contributing equal total weight shared
by window duration. Validation alone chooses a threshold by balanced accuracy.
The preserved baseline uses its original threshold. Reports include balanced
accuracy, synthetic F1, FPR/FNR, confusion counts and speaker breakdowns.
Single-class speaker rows have null balanced accuracy and undefined rates.
Diagnostic microphone labels come from the previous audit/user attestation;
those clips never influence model fitting, threshold or configuration choice.

Numeric libraries use one thread. Feature extraction is bounded by nine minutes
at most and reports progress every 100 clips; a timeout retains caches for a
resume. Preparation has a five-minute limit. Logistic regression is limited to
1,000 iterations and a fit exceeding 120 seconds is not exported. These are
stage limits, not a promise about total wall-clock time. Reports and models use
exclusive creation; choose a fresh experiment/model path for a new fit.
Data and reports live under the existing ignored `data/` tree. There are no
Luminal calls, cloud inference dependencies or frontend changes.

The candidate is exported separately as `models/inthewild-v1-12000.joblib`.
The baseline remains the app default. After explicit deployment approval,
selection uses the app's existing mechanism (restart required):

```sh
ECHOPROOF_MODEL_PATH="$PWD/models/inthewild-v1-12000.joblib" .venv/bin/streamlit run app.py
```

Rollback:

```sh
ECHOPROOF_MODEL_PATH="$PWD/models/baseline.joblib" .venv/bin/streamlit run app.py
```

Model loader support for the new artifact validates feature settings, label
mapping, pinned provenance and disjoint training/validation speakers. The
stable inference API and uncalibrated score semantics are unchanged. Validation
improvement alone does not authorize deployment. A single genuine diagnostic
clip cannot establish microphone reliability, and there is no fresh external
synthetic set in this experiment. Assess both classes on new microphones,
languages, codecs and generators before any generalization claim.
