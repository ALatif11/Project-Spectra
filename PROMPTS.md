# EchoProof AI — Copy-paste prompts for a two-person team

**Order:** First run Prompt A (Codex) and Prompt B (Claude) in parallel on separate branches/worktrees. Next run C (Codex) and D (Claude). Then E (Codex) and F (Claude); G is optional. Finish with H. Each prompt is one bounded work unit: do not send all prompts at once.

## Prompt A — You / Codex: bootstrap the real backend

> You own the EchoProof backend. Follow AGENTS.md. Work only in `src/`, `tests/` for backend, and initial shared bootstrap files (`requirements.txt`, `.gitignore`, `.env.example`, `README.md`) if they do not exist. Do not create or edit `app.py` or `ui/` because a teammate owns them.
>
> **This task only:** create Python 3.11 project setup, safe local audio decoding from bytes, mono/16kHz preprocessing, 4-second windows, log-Mel spectrogram extraction, and a compact MFCC/spectral-features extractor using Librosa/NumPy. Add narrow pytest tests using programmatically generated tone/silence samples for decoding, shape, invalid audio, and consistent deterministic features. Do not train a model or integrate Luminal yet.
>
> Publish functions with clear types and docstrings. Do not download large datasets. Do not run any command taking >60 seconds without approval. Run targeted tests, fix failures, and report filenames changed plus the exact function signatures your teammate may call. Stop when this milestone passes.

## Prompt B — Teammate / Claude Code: parallel UI skeleton

> You own EchoProof's interface. Follow AGENTS.md. Work ONLY on `app.py`, `ui/`, `assets/` and frontend tests; do not edit `src/`, `agents/`, `database/`, or shared dependency files. Another developer is building the backend in parallel.
>
> **This task only:** implement a polished dark Streamlit app with an audio upload area, audio player, waveform and log-Mel spectrogram panels, and a detector result area. Include tabs called **Audio Detective**, **Agent Lab**, and **Model Evolution**, but make the last two honest empty states for now.
>
> Integrate through `src.inference.predict_audio(audio_bytes, filename=None)` using the fixed response schema in AGENTS.md. Since the backend might not exist, catch ImportError and show 'Detector not available yet' rather than fake predictions. If mock fixture data is needed for UI layout tests, visibly label it 'UI demo fixture — not a model result', keep it separate from live inference, and default to real mode.
>
> No invented confidence numbers or metrics. Use a minimal package footprint; if additional dependencies are needed, list them for the backend owner instead of changing requirements.txt. Add one smoke test if practical. Run the app or a Python import smoke check. Report file changes and tests; stop.

## Prompt C — You / Codex: real training and inference

> Follow AGENTS.md and preserve the frontend ownership boundary. Implement a real dataset loader and reproducible training/evaluation workflow in `src/` plus appropriate tests. Prefer ASVspoof 2019 Logical Access with explicit bonafide/spoof labels and official train/development/evaluation protocol. Provide a documented CLI for a *small, bounded* training subset; never download the full dataset automatically. If real labeled audio is absent, make the trainer explain how to obtain/configure it and exit cleanly; do not fabricate trained checkpoints or results.
>
> Train a fast scikit-learn pipeline (scaler fitted on train only + logistic regression) from spectrogram-derived MFCC/spectral features. Persist a real versioned model checkpoint, feature settings, selected validation threshold and metadata. Write `src/inference.py` with the EXACT `predict_audio(audio_bytes: bytes, filename: str | None = None) -> dict` contract in AGENTS.md. Return `unavailable` when no model exists; `invalid` for bad audio; no random or hard-coded detection values. Return per-segment synthetic scores. Evaluate only with held-out labels and save real metrics as JSON. Add and run narrow tests. Do not edit `app.py` or `ui/`.
>
> Completion output: commands to train, evaluate, test inference, and precise model/threshold limitations. Stop before building Luminal agents.

## Prompt D — Teammate / Claude Code: wire backend and visualize

> Follow AGENTS.md. The backend now exposes `src.inference.predict_audio`. Work only in frontend-owned files. Wire real upload bytes to that exact API. Use local UI-side waveform/spectrogram computation as needed, or call existing backend visualization-friendly functions if documented; DO NOT duplicate training or invent model scores.
>
> Show the response label, raw synthetic score, decision threshold, segment timeline, and an inconclusive/unavailable error state. Never call the numeric score calibrated confidence unless the backend explicitly documents calibration. Make dark-mode layout responsive. Ensure fresh uploads rerun inference rather than showing stale cached results. Add concise warnings that predictions aren't forensic proof. Run an app import/smoke test and record the results. No Luminal or MongoDB integration yet.

## Prompt E — You / Codex: bounded Luminal self-improvement loop

> Follow AGENTS.md. Implement the Luminal agent experiment extension only after the baseline trainer and inference work. Limit changes to `agents/`, backend-owned code, and backend tests. No AWS and no broad ML framework changes.
>
> Implement `luminal_client.py` with configurable credentials, base URL, model ID, timeouts, and explicit, informative errors. Do not guess a Luminal endpoint or model ID; consult actual available API docs/config or leave placeholders for environment variables. Create three **distinct** proposal roles: features, classifier, robustness. Each produces structured JSON constrained to an allowlisted experiment schema; reject invalid settings. The LLM may *propose* experiments but cannot execute arbitrary generated Python/shell. A deterministic Python runner fits allowed scikit-learn candidates, computes real validation metrics on fixed partitions, and records seed, parameters, status, runtime, and accepted/rejected decision. It must not access final test labels during optimization.
>
> Add hard caps: maximum 3 rounds, 3 agents/round, 20 LLM calls, 12 model experiments, 1000 output tokens/call when supported, and configurable train/run timeout. Provide a `--dry-run` mode using clearly marked static proposals to test orchestration without cloud calls, and a genuine cloud mode which never fabricates responses. Output a readable `runs/experiments.jsonl` (ignored by Git) and a clean redacted example schema for frontend integration. Demonstrate actual local experiments if labeled data is available; otherwise report exactly what couldn't be run. Run targeted tests and stop.

## Prompt F — Teammate / Claude Code: live experiment dashboard

> Follow AGENTS.md and edit only frontend-owned files. Build the Agent Lab and Model Evolution pages from recorded experiment events. Use the actual schema in `runs/experiments.jsonl` or the safe loader the backend owner documents. Show feature/model/robustness agent proposal cards, real statuses, settings, metrics, and an experiment history table.
>
> Plot validation F1 or balanced accuracy across completed runs, marking the selected best model. Include failed, rejected, and no-improvement rounds. Never invent successful trajectories. If the file does not exist, display a useful empty state. Clearly label dry-run fixtures. Recheck that the Audio Detective tab still uses local inference and doesn't send audio to cloud services. Test import/render where practical. Do not modify experiment runner files.

## Prompt G — You / Codex: optional MongoDB memory

> Follow AGENTS.md. This is optional: only start after the detector and Luminal experiment loop work. Implement a small repository abstraction for experiment events that writes to MongoDB if `MONGODB_URI` is configured, otherwise falls back to append-only local JSONL. Preserve the exact experiment JSON schema consumed by the UI. Store configurations, scores, status, timestamps, and de-identified failure categories, but never raw uploaded user audio or secrets. The agent system should be able to read recent past results before proposing another experiment. Add tests for the fallback and error handling. Do not touch UI-owned files or require MongoDB for core inference.

## Prompt H — Both teammates: final integration and demo

> Follow AGENTS.md. We are approaching hackathon judging. Make only high-impact, narrowly scoped fixes within your owned files. Do not add features. Verify these in order: (1) real recorded or labeled input uploads; (2) waveform/spectrogram renders; (3) a **real trained** detector gives an output, or clearly says unavailable; (4) agent loop shows actual proposals and executed experiment metrics, including failures; (5) local inference still works while offline; (6) invalid/silent input fails gracefully; (7) no API keys or large audio data in Git. Run tests that finish quickly. Report any incomplete capabilities accurately and hand off a 90-second demo sequence. No fake evaluation results or silent workarounds.

## Coordination and usage tips

- After Prompt A and B complete, share the exact `predict_audio` contract and `requirements.txt` changes before merging.
- Use a capable model for Prompts C/E and a faster model for tiny UI polish, tests, and docs; avoid high-effort reasoning by default.
- For follow-up bugs, paste the traceback plus the specific file, and ask for a targeted fix only.
- Commit a working milestone before starting the next prompt.
- Before judging, prioritize a reproducible local live demo over optional AWS or a cloud deployment.
