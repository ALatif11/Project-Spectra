# EchoProof AI — Repository Instructions

## Mission and priorities
Build a working, honest hackathon MVP for audio deepfake screening: locally processed audio, a log-Mel spectrogram, a trained real-vs-synthetic classifier, and optional Luminal-powered agents that run measurable model experiments. Prioritize: (1) working inference, (2) trustworthy evaluation, (3) clean demo, (4) bounded Luminal agent loop, (5) MongoDB, (6) optional AWS. Do not add cloud dependencies to core prediction.

## Stack and layout
- Python 3.11; Streamlit; NumPy, Librosa, SoundFile; scikit-learn; Plotly; pytest.
- `app.py`, `ui/`: frontend (Claude Code teammate).
- `src/`: audio decoding, spectrograms, features, datasets, training, evaluation, local inference (Codex teammate).
- `agents/`: Luminal client, constrained proposals, experiment runner (Codex teammate).
- `database/`: optional MongoDB adapter and JSONL fallback (Codex teammate).
- `tests/`: each teammate owns tests for their own modules.
- `models/`, `data/`, `runs/`: local artifacts, excluded from Git where appropriate.
- `README.md`, `requirements.txt`, `.env.example`, `.gitignore`: shared files; coordinate before editing.

## Parallel development and file ownership
- Codex owns `src/`, `agents/`, `database/`, model/data scripts, and their tests.
- Claude Code owns `app.py`, `ui/`, visualization and frontend tests.
- Neither edits the other's files without explicit teammate agreement. Never overwrite working changes or reset the other person's branch.
- Work on separate branches (`backend-ml`, `frontend-ui`) or separate worktrees. Integrate at defined checkpoints, preferably through pull requests.
- Before changing a shared dependency or interface, communicate the exact change to the other teammate.
- Preserve existing code; use small diffs and targeted fixes. Never refactor unrelated modules.

## Stable inference contract (do not change without agreement)
Backend provides in `src/inference.py`:

```python
def predict_audio(audio_bytes: bytes, filename: str | None = None) -> dict:
    """Run local audio inference; do not call a remote API."""
```

Return a JSON-serializable dict with exactly these required top-level keys:

```json
{
  "status": "ok",
  "label": "likely_human",
  "synthetic_score": 0.12,
  "threshold": 0.50,
  "segments": [{"start_s": 0.0, "end_s": 4.0, "synthetic_score": 0.12}],
  "model_version": "baseline-v1",
  "message": "Research-grade estimate; not forensic proof."
}
```

- Allowed status values: `ok`, `invalid`, `unavailable`.
- Allowed labels: `likely_human`, `likely_synthetic`, `inconclusive`.
- `synthetic_score` and `threshold`: floats in [0,1] when available, otherwise `null`; these are model outputs/decision thresholds, **not calibrated probabilities** unless calibration was verified.
- For `invalid` or `unavailable`, use `inconclusive`, empty `segments`, and an actionable `message`.
- Each segment has seconds-based `start_s`, `end_s`, `synthetic_score`.
- Do not rename keys or silently change score semantics. Document any optional fields.
- Audio must be decoded and preprocessed entirely locally; use mono, 16 kHz and consistent training/inference windows; handle silence and malformed uploads.
- UI can display labeled **mock UI fixtures** while backend is incomplete, but must never present mock scores as actual predictions; show `unavailable` until a real model exists.

## Dataset: Gary Stafford Hugging Face v4 (authoritative for this project)
- Dataset repo: `garystafford/deepfake-audio-detection` at https://huggingface.co/datasets/garystafford/deepfake-audio-detection (CC BY 4.0, acknowledge author in README).
- Public dataset of 1,866 English-language FLAC clips (16 kHz mono): 933 genuine, 933 synthetic; 2.5–13 s long. Synthesized speech includes six TTS vendors. **Do not use ASVspoof as the default dataset.**
- The Hugging Face `datasets` view exposes a **single `train` split**. There are NO pre-supplied validation/test partitions. We must create persistent local splits (e.g., `data/splits.csv`), not treat the displayed `train` as already separated.
- Prefer raw `real/*.flac` and `fake/*.flac` directories for a lightweight loader and reliable original filenames. Implement `scripts/download_dataset.py` using `huggingface_hub.snapshot_download(repo_id=..., repo_type='dataset', allow_patterns=['real/*.flac','fake/*.flac','README.md'], local_dir='data/deepfake-audio-detection')`. Require explicit invocation of this substantial download and handle network failures; do not automatically download on importing or starting Streamlit.
- Label map MUST be verified against directory/actual features, never inferred from alphabetical order: `real/ => 0` and `fake/ => 1`. If using HF Parquet, inspect `dataset.features['label']` and check actual mapping instead of trusting a label-order assumption.
- **Critical leakage warning:** genuine segments come from only 14 original YouTube videos. Filenames like `yt_0000_part_001.flac` and `yt_0000_p2_part_167.flac` are from the same recording, and must stay in ONE partition. Synthetic segments like `el_0001_part_001.flac` and `el_0001_c_part_002.flac` should likewise be grouped by their source ID. Implement explicit grouping using inspected filenames, not simply a random row split. Verify groups do not cross split boundaries. If source grouping cannot be recovered, state limitation and don't report random-split accuracy as independent generalization.
- Use a seeded group-aware train/validation/test split (target ~70/15/15 by examples, exact class proportions need not be perfect). Check each partition includes both classes, persist membership and source/group ID, fit scalers only on training, select thresholds only on validation, and use untouched test only once at the end.
- Show grouped held-out evaluation and, if feasible, a second robustness check using a distinct TTS vendor, holding that vendor out of optimization. The latter may require a separate experiment split. Never claim unseen-generator generalization if it was trained/optimized on that generator.
- Start with MFCC + spectrogram-derived statistics and a scikit-learn pipeline (`StandardScaler` + logistic regression). Do not use a large pretrained model or CNN as the initial dependency. Cache features after extraction; rerun lightweight model fits for agent experiments. The published dataset's optional Wav2Vec2 model was trained on this dataset and should not be treated as independent benchmark evidence.
- This is a small English research dataset with real-recording source bias; a classifier may learn YouTube recording artifacts versus clean TTS rather than synthetic speech itself. State uncertainty and measure domain shift; no predictions are forensic proof.

## Data, modeling, and honesty
- Preserve original provenance and license attribution. Keep dataset and caches out of Git by default.
- Train on genuine labeled recordings only; avoid duplicate clips, overlapping source IDs, and leakage; make seeds explicit. No training on untouched final test.
- Record real metrics and experiments. Never invent model accuracy, classification outputs, chart data, model checkpoints, or agent improvements.
- A score is not proof of authenticity. Provide an inconclusive state, plain-language limitations, and explain likely domain shift on unseen generators.
- Do not claim Grad-CAM regions are definitive signs of manipulation.

## Luminal agent budget and safety
- Cloud LLM agents propose experiments; deterministic Python executes/validates training and metrics. Agents never execute arbitrary generated source or shell commands.
- Proposals must use a whitelisted JSON schema, bounded hyperparameters, allowed feature sets, and explicit compute constraints.
- Default limits: 3 rounds, <=3 agents/round, <=20 LLM calls total, <=12 model fits, <=1000 generated tokens/call (if API supports this), and configurable timeouts.
- Respect endpoint/model names actually shown in Luminal docs or dashboard. Never invent an API URL or assert a particular model is available.
- Never send user-recorded raw audio to Luminal; send summary metrics and anonymized metadata only.
- Save proposals, failures, actual metrics, seeds, and chosen configurations. Unsuccessful experiments are meaningful data. Do not claim improvement without measured validation gains.

## Secrets, resources, and scope
- Keep secrets in `.env` or host-provided environment variables; commit only `.env.example`. Never print real API keys.
- Do not upload private raw audio or samples to MongoDB/AWS without explicit opt-in.
- Avoid adding AWS, MongoDB, Docker, or new ML frameworks until the local detector and demo are functional.
- No unbounded agent loops, broad dependency upgrades, recursive re-generation, large automatic downloads, or lengthy training by default; ask before expensive operations. Download the dataset only when the user explicitly runs/approves the download command.
- Large datasets/checkpoints, `.env`, `runs/`, `data/`, `models/*.joblib`, `models/*.pt` belong in `.gitignore` unless an intentionally tiny public fixture is approved.

## Working method and definition of done
1. Inspect only task-relevant files; make the smallest working change.
2. Run the narrowest tests relevant to the change, then `python -m pytest -q` when practical.
3. Keep imports compatible with running `streamlit run app.py` from repository root.
4. Report exactly what changed, tests run, failures, and next integration dependency in <=10 lines.
5. Stop at the requested milestone; do not autonomously implement the entire roadmap.

An MVP is done when a legitimate held-out-labeled audio example can be processed locally and the UI shows a real model result and spectrogram, including invalid/unavailable handling. The multi-agent prize extension is done only when competing actual agent proposals, executed experiments, and measured results can be replayed on the dashboard.
