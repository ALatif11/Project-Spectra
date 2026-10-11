# EchoProof

**Check a voice clip before you act on it.**

Voice cloning now needs only a few seconds of someone's audio, and "family emergency" calls with a cloned voice are a known scam. EchoProof lets an everyday person drop in a voice note or call recording and get a plain answer in seconds: does this sound closer to a real human voice or to AI-generated speech, *why*, and what to do next.

Everything runs on your own computer. The audio is never uploaded.

> EchoProof is a research prototype built for a hackathon. Its score is a signal, not proof. See [Limitations](#limitations).

## What it does

- **Plain answer.** "This clip sounds closer to AI-generated speech", "...closer to a real human voice", or "We can't tell from this clip".
- **Score with context.** A meter from human-like to AI-like, the model's decision threshold, and the "can't tell" zone around it.
- **Why.** The measurements of the recording that moved the score most (for example loudness changes or voice tone), read directly from the model.
- **Clip-quality warnings.** Very short clips, lots of silence or distorted audio are flagged as harder to judge.
- **What to do next.** Verification steps based on FTC guidance (call back on a known number, use a family code word, treat payment requests as the warning sign), plus where to report fraud.
- **Save report.** A downloadable summary of the check.
- **Model transparency.** A Model page showing which checkpoint is running (with its SHA-256 fingerprint) and the replayable history of the AI-agent experiments used to improve it.

## Quick start

Requires **Python 3.11** (3.12 also works for the app).

```bash
git clone https://github.com/ALatif11/Project-Spectra.git
cd Project-Spectra
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install the backend and app dependencies:

```bash
python -m pip install -r requirements-backend.txt streamlit plotly
```

### Get a model

The app needs a trained checkpoint at `models/baseline.joblib`. Model files are not stored in Git. Either copy it from a teammate (only accept `.joblib` files from people you trust; they are Python pickles) or train it yourself:

```bash
python -m scripts.download_dataset
python -m scripts.prepare_dataset --seed 42
python -m scripts.train_model
```

The download is about 570 MB. Details: [src/README.md](src/README.md).

### Run the app

```bash
python -m streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Without a model the app still opens and says the detector is unavailable.

### Run the In-the-Wild demo model

The team's newer model, retrained on the In-the-Wild dataset, is launched with a script that verifies the frozen checkpoint's fingerprint before starting the app:

```bash
python -m scripts.launch_final_demo
# roll back to the preserved baseline
python -m scripts.launch_final_demo --rollback
```

This needs `models/inthewild-final-frozen.joblib`, `models/baseline-rollback-20261011.joblib` and the freeze record in `runs/inthewild-final-20261011/`, which are produced locally and not stored in Git. Without them the script stops instead of silently using another model. To run any checkpoint by hand, set `ECHOPROOF_MODEL_PATH` to its path before starting Streamlit.

## How it works

1. **Decode locally.** The file is decoded in memory and converted to 16 kHz mono (`src/audio.py`).
2. **Four-second chunks.** Each chunk gets 52 acoustic measurements: MFCC averages and variation plus brightness, bandwidth, rolloff, flatness, zero-crossing rate and loudness (`src/features.py`).
3. **Score.** A scaler and logistic-regression classifier score each chunk; the clip score is the duration-weighted average. Scores within 0.05 of the threshold are reported as "can't tell" (`src/inference.py`).
4. **Explain.** The "Why" list is computed from the loaded model's own weights for this clip (`ui/explain.py`). It hides itself for model types that don't expose weights.

### AI agents (Luminal)

Three agents (feature, classifier and robustness roles) propose model changes through Luminal's OpenAI-compatible API. Deterministic Python trains and measures every proposal locally and keeps it only if it beats the current best on validation data. The agents only see summary metrics, never audio. Budgets are hard-capped in code. Every proposal, failure and metric is logged to `runs/experiments.jsonl` and replayed on the Model page.

```bash
python -m pip install -r agents/requirements.txt
cp .env.example .env   # then add your LUMINAL_API_KEY; .env is git-ignored
python -m scripts.test_luminal
python -m agents --dry-run   # offline, static proposals, real local fits
python -m agents             # live Luminal run
```

Details: [agents/README.md](agents/README.md) and [agents/research.md](agents/research.md).

## Results

From the reports committed in `runs/`, on the original [Gary Stafford deepfake audio dataset](https://huggingface.co/datasets/garystafford/deepfake-audio-detection) with speaker-grouped splits:

| Model | Validation balanced accuracy | Final test balanced accuracy (evaluated once) |
|---|---|---|
| Baseline | 95.11% | 96.64% |
| Agent-optimized (Luminal run `ba52f0c8`) | 95.68% | 96.42% |

The agents' pick improved validation but **not** the final test, so the baseline stays the default. Full write-up: [src/deployment.md](src/deployment.md).

**In-the-Wild retraining.** The team reports 92.92% validation balanced accuracy for the retrained model versus 46.45% for the original model on the same In-the-Wild validation set. This is cross-dataset adaptation from new training data, not agent-driven improvement. Its report files are not in the repository yet, so these numbers can't be verified from this repo. Workflow: [src/inthewild_experiment.md](src/inthewild_experiment.md).

## Limitations

- **Scores are uncalibrated.** A model score is not a probability, not a confidence level and not forensic proof.
- **Domain shift.** The original training data is narrow (one English dataset). Genuine speech recorded on unfamiliar microphones, in echoey rooms, in other languages or with long pauses can score as AI-like. A real microphone recording scored 0.999 on the original baseline.
- **Unseen generators.** New voice-cloning tools the model has never heard may score as human-like.
- **Always verify another way.** EchoProof is one signal. Call the person back on a number you already know before sending money or information.

## Privacy

Audio is processed in memory on the local machine and is never uploaded. (The page itself loads its fonts from Google Fonts.) The optional Luminal agents send only aggregate metrics and allow-listed settings, never audio, file names or per-clip scores.

## Project layout

| Path | What |
|---|---|
| `app.py`, `ui/` | Streamlit app: check-a-clip flow, explanations, Model page |
| `src/` | Decoding, features, training, evaluation, local inference (`predict_audio`) |
| `agents/` | Luminal client, constrained proposals, experiment runner |
| `scripts/` | Command-line entry points for data, training, evaluation and demo launch |
| `tests/` | pytest suite |
| `runs/` | Experiment history and evaluation reports (most new runs are git-ignored) |
| `models/`, `data/` | Local checkpoints and datasets (git-ignored) |

## Tests

```bash
python -m pytest -q
```

Known issues on Windows: `tests/test_agents.py::test_failure_diagnostics_keep_only_safe_metadata` exceeds the Windows environment-variable size limit, and the In-the-Wild feature extraction imports the Unix-only `resource` module.

## Data and credits

- Gary Stafford, [Deepfake Audio Detection Dataset (v4)](https://huggingface.co/datasets/garystafford/deepfake-audio-detection), CC BY 4.0.
- [In-the-Wild](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/InTheWild) (Apache-2.0 redistribution). Müller et al., *Does Audio Deepfake Detection Generalize?*, Interspeech 2022.
- Scam guidance: [FTC consumer alert on AI voice cloning](https://consumer.ftc.gov/node/80688). Report fraud at [ReportFraud.ftc.gov](https://reportfraud.ftc.gov/).
