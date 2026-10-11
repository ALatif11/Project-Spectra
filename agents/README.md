# Bounded local experiments with optional Luminal proposals

This extension requires the locally trained Gary Stafford baseline and its
unchanged `data/splits.csv`. Every run first reproduces baseline validation
metrics and verifies actual local inference. Nothing downloads on import.
`src/inference.py` keeps the stable prediction interface and has no cloud dependency.

From the repository root, using the existing backend Python environment:

```sh
python -m agents --dry-run
python -m agents --dry-run --rounds 1 --train-timeout 60 --run-timeout 180
python -m agents
```

`--dry-run` means **static offline proposal fixtures**, with **actual local fits
and measured validation metrics**. It makes zero cloud calls and is labeled
`mode="dry_run_static_proposals"`, `dry_run=true` in every event. It is not
evidence of LLM/agent reasoning. Cloud mode uses only actual provider responses;
missing configuration or failed requests never fall back to static proposals.

## Verified Luminal configuration

The dashboard confirms the OpenAI-compatible API at
`https://flash.luminal.cloud/v1`, model `deepseek-ai/DeepSeek-V4.1-Flash`, and
Bearer authentication. All three roles share the official OpenAI Python SDK
client and `client.chat.completions.create()`. Optional dependencies are listed
in `agents/requirements.txt`; core audio inference imports none of them.

The client loads `.env` from the repository root automatically with
python-dotenv, regardless of working directory. Host environment variables take
precedence. `.env.example` lists `LUMINAL_API_KEY`, `LUMINAL_BASE_URL`, and
`LUMINAL_MODEL_ID`, plus bounded budgets. Keep the real key only in ignored
`.env`. No shell sourcing or exports are needed. The former completion path,
API format, and token field settings are removed.

Chat requests use `max_tokens`, with `MAX_OUTPUT_TOKENS=1000` as a hard ceiling.
If Luminal rejects the field, the request fails safely; there is no automatic
retry or parameter fallback. SDK retries are disabled with `max_retries=0`.
Redirects are refused. `LUMINAL_TIMEOUT_S` defaults to 30 seconds (maximum 120).
Cloud requests run in killable processes to respect the overall run timeout.

`scripts/test_luminal.py` makes exactly one small chat request, requesting at
most 16 output tokens. It logs only success or a redacted error, never the key
or response. `--dry-run` checks configuration with zero API requests. This test
does not load audio, require the dataset/model, or run training.

Only aggregate train/validation counts, metrics, accepted allowlisted settings,
and redacted failure categories leave the computer. No audio, original filenames,
paths, source IDs, per-clip scores, test labels, or credentials appear in prompts
or experiment logs. Provider response bodies are discarded on errors.

## Proposals and deterministic execution

- **features:** change the feature subset while fixing baseline logistic settings.
- **classifier:** tune bounded logistic regression or random forest using all features.
- **robustness:** balanced logistic regression with level-related features removed.

`proposal.schema.json` is the exact allowlisted JSON structure. Additional fields,
role mismatches, non-finite numbers, oversized/ambiguous JSON, seed overrides,
unbounded settings, and executable code are rejected. Nothing evaluates generated
Python or runs generated shell commands. `no_energy` removes RMS statistics and
MFCC coefficient 0 statistics; this reduces some level cues but does not remove
all recording bias. Random forests use one worker and bounded tree count/depth.

All candidates use the shared four-second preprocessing/features cache. The
scaler and classifier fit only retained training windows, with unit total weight
per clip, and the seed persisted in the fixed split file. The optimization reader
discards final test labels and exposes only train/validation rows. Neither final
test audio nor final test metrics are read. Both the split file and baseline
artifact are hashed and checked before/after every experiment. No assignments
are generated or rewritten by the agent system. Duplicate configurations are
rejected without fitting.

The decision threshold maximizes validation clip balanced accuracy, with the
baseline tie rule. A candidate is `accepted` only if its measured validation
balanced accuracy strictly exceeds the incumbent; ties and regressions are
`rejected`. The incumbent starts at the reproduced baseline. Repeated validation
optimization can overfit this small partition; these metrics are not independent
test performance or calibrated probabilities. The synthetic-vendor recall and
real specificity diagnostics are **seen-vendor validation subsets**, not an
unseen-generator holdout or proof of domain-shift robustness.

The selected configuration is recorded as `best_proposal` in each event and the
final run event. No candidate replaces `models/baseline.joblib` or changes local
inference. Selection identifies a configuration for later reviewed promotion and
one-time final evaluation; this extension does not evaluate final test itself.

## Hard budgets and replay

Hard upper bounds are 3 rounds, 3 roles per round, 20 total cloud-call attempts,
12 model-fit attempts, and at most 1000 requested output tokens per call.
Failures/timeouts consume their attempt budgets. With one proposal per role per
round the current loop makes at most 9 calls/fits, inside those hard bounds.
Root `.env` values `MAX_AGENT_ROUNDS`, `MAX_LLM_CALLS`, and
`MAX_MODEL_EXPERIMENTS` set the default budgets within those hard bounds.
`MAX_OUTPUT_TOKENS` may also be lowered. CLI options can **lower** configured
call/fit/round caps but cannot raise them:
`--max-llm-calls`, `--max-experiments`, `--rounds`, `--agents-per-round`.
`--train-timeout` (default 120 seconds) bounds each baseline check/candidate;
`--run-timeout` (default 600 seconds) also includes startup and cloud requests.
Workers are terminated on timeout; cleanup may add up to one second.

`runs/experiments.jsonl` is already covered by the repository's `runs/` ignore
rule. It is append-only across runs. Each line is one UTF-8 JSON event with a
unique `run_id`; startup, baseline, rejected/failed/completed experiments, and
final selection are replayable. New events include the effective `limits` for
replay (this field is optional for earlier v1 events). A failed run exits nonzero and records why.
The optional SDK dependencies are isolated in `agents/requirements.txt`.
No frontend changes, AWS, or new ML frameworks are required.

Frontend integration uses `event.schema.json` and `example_event.json`. The
example is explicitly a **documentation fixture**, with null measurements and
redacted identities, never an actual prediction or measured improvement.
Group events by `run_id`; use `event="experiment"` for cards/history and the
`run_finished` event's `best_proposal` for the selected configuration. Show
`status`, `decision`, and `reason` for failures, rejection, and no improvement.
Plot only real, non-null `metrics` from `status="ok"`; label static-proposal runs
clearly. Metrics are always from the fixed validation partition. The schema
does not expose any user recordings. A missing event file should show an empty
state. Frontend code remains owned by the frontend teammate.

Tests: `python -m pytest -q tests/test_dataset.py tests/test_model.py tests/test_agents.py`.
