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

## Data, modeling, and honesty
- Use genuine labeled bonafide/spoof recordings with provenance and permitted use. Prefer ASVspoof 2019 Logical Access official train/development/evaluation protocol. Do not train on its evaluation split.
- Avoid speaker, source, or duplicated-clip leakage; make seeds and split strategy explicit. Use training data for fitting transforms, development data for model selection, and untouched evaluation data for the final report.
- Start with fast MFCC/spectral-feature extraction + scikit-learn logistic regression; add CNN/AASIST only after end-to-end MVP works.
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
- No unbounded agent loops, broad dependency upgrades, recursive re-generation, massive automatic downloads, or lengthy training by default; ask before expensive operations.
- Large datasets/checkpoints, `.env`, `runs/`, `data/`, `models/*.joblib`, `models/*.pt` belong in `.gitignore` unless an intentionally tiny public fixture is approved.

## Working method and definition of done
1. Inspect only task-relevant files; make the smallest working change.
2. Run the narrowest tests relevant to the change, then `python -m pytest -q` when practical.
3. Keep imports compatible with running `streamlit run app.py` from repository root.
4. Report exactly what changed, tests run, failures, and next integration dependency in <=10 lines.
5. Stop at the requested milestone; do not autonomously implement the entire roadmap.

An MVP is done when a legitimate held-out-labeled audio example can be processed locally and the UI shows a real model result and spectrogram, including invalid/unavailable handling. The multi-agent prize extension is done only when competing actual agent proposals, executed experiments, and measured results can be replayed on the dashboard.
