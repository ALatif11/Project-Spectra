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

> Follow AGENTS.md and preserve frontend ownership. Replace ASVspoof references with the public Gary Stafford dataset: `garystafford/deepfake-audio-detection` (v4) at https://huggingface.co/datasets/garystafford/deepfake-audio-detection. It has 1,866 16-kHz mono FLAC clips, with `real/` (933) and `fake/` (933), and only ONE Hugging Face `train` split. **It has no official held-out test split.**
>
> Implement `scripts/download_dataset.py` using `huggingface_hub.snapshot_download` to retrieve only `real/*.flac`, `fake/*.flac`, and `README.md` into `data/deepfake-audio-detection/`, but don't download during module import, app startup, or without an explicit user command. Avoid duplicating Parquet and raw audio. Map folder labels `real=0` / `fake=1`, and validate this against inspected sample metadata if reading the Parquet alternative. Detect and report corrupt or missing clips.
>
> Crucial: genuine audio was split from only 14 YouTube sources. Group all `yt_0000_*` samples together, and likewise group synthetic chunks with their source prefix and ID, e.g., `el_0001_*`. Inspect actual filenames before implementing regex. Create persistent seeded GROUP-AWARE train, validation, and untouched test partitions, target ~70/15/15 with both classes in every partition. Save `data/splits.csv`; test that no original source/group is shared across partitions. Never use random row-level train_test_split on segments. If grouping is incomplete, explain limitations. Optionally measure one held-out TTS-vendor robustness test separately.
>
> Reuse existing audio feature functions and train a fast scikit-learn `StandardScaler` + `LogisticRegression` model on spectral/MFCC features. Fit scaler only on training, select threshold only on validation. Persist model and feature settings. Implement exact `src.inference.predict_audio(audio_bytes: bytes, filename: str | None = None) -> dict` contract from AGENTS.md, with local segment inference and real scores. Save actual validation and final test metrics. Add tests for downloader manifest, class mapping, grouping, no leakage, saved model, and invalid audio. No mocked metrics or checkpoints. Edit backend-owned files only.
>
> Give me exact commands to download, prepare, train, evaluate, and locally test inference. Stop after this milestone passes. No Luminal integration yet.

## Prompt D — Teammate / Claude Code: wire backend and visualize

> Follow AGENTS.md. The backend now exposes `src.inference.predict_audio`. Work only in frontend-owned files. Wire real upload bytes to that exact API. Use local UI-side waveform/spectrogram computation as needed, or call existing backend visualization-friendly functions if documented; DO NOT duplicate training or invent model scores.
>
> Show the response label, raw synthetic score, decision threshold, segment timeline, and an inconclusive/unavailable error state. Never call the numeric score calibrated confidence unless the backend explicitly documents calibration. Make dark-mode layout responsive. Ensure fresh uploads rerun inference rather than showing stale cached results. Add concise warnings that predictions aren't forensic proof. Run an app import/smoke test and record the results. No Luminal or MongoDB integration yet.

## Prompt E — You / Codex: bounded Luminal self-improvement loop

> Follow AGENTS.md. Implement the Luminal agent experiment extension only after the baseline trainer and inference work on the Gary Stafford dataset. Reuse the fixed group-aware train/validation splits from `data/splits.csv` in EVERY experiment; never train on held-out test groups, and never change group assignments to inflate validation accuracy. Limit changes to `agents/`, backend-owned code, and backend tests. No AWS and no broad ML framework changes.
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

## Quick dataset-switch instruction (send NOW if Codex has already started on ASVspoof)

> We are switching to `garystafford/deepfake-audio-detection` (Hugging Face, v4). Read updated AGENTS.md. Preserve completed audio preprocessing, spectrogram, features, and the fixed `predict_audio` API. Replace only dataset-specific assumptions, data download/loading, split generation, training instructions, and tests. Dataset is public; 1,866 16kHz FLAC samples in real/fake folders. No official validation/test split; create reproducible grouped splits and keep source chunks from the same YouTube video or synthetic source in one split. Label `real=0`, `fake=1` explicitly. Generate a one-time raw FLAC downloader script and exact commands, but do not auto-download on Streamlit startup. Ensure dataset files/caches are gitignored. Don't touch frontend files, Luminal settings or secrets. Run tests and provide the precise next command I should execute.
