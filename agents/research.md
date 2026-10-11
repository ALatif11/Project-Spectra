# Expanded InTheWild research profile

This extends the existing orchestrator, Luminal client and three specialized
roles. It adds no cloud inference, agent framework, dataset download or neural
model. Configuration and ordinary preflight make zero paid calls and zero fits.
Real local benchmarks and subsequent live execution need separate approval.

## Code-enforced bounds

`ResearchPolicy` validates at most six rounds, exactly three roles, 60 API
attempts, 36 actual ML fit attempts, 2,500 requested output tokens, $5 estimated
spending and 900 seconds of run time. A local operation is bounded by 120 seconds,
and a provider attempt by 60 seconds by default. Lower limits can be configured;
upper ceilings cannot be raised through JSON, environment or CLI configuration.
SDK retries, parameter fallbacks and cloud-failure offline fixtures are absent.
A request/fit exceeding remaining time is terminated by the existing bounded
worker. Cleanup/startup is separate from the measured research budget.

`agents/research_profile.json` is the explicit profile. The repository-root .env,
legacy profile defaults and original frontend event schema are preserved. The
new profile deliberately uses its own recorded budgets rather than inheriting
the older three-round/1,000-token settings. Luminal credentials still come only
from the existing environment loader. Tokens are requested through the existing
OpenAI-compatible `max_tokens` field. Provider acceptance of 2,500 has not been
tested with a paid probe; HTTP 400/422 stops the live run, without parameter
fallback or retry.

The three-role/one-proposal-per-role schedule can issue at most **18** requests.
The 60-call ceiling is enforced as an absolute limit, not a spending target.
Each configuration needs three development fits; a fourth, final full-training
fit is reserved only for the selected winner. With a prepared baseline cached,
36 fit slots support at most 11 newly evaluated configurations plus that final
fit (**34 actual fits**). If the development baseline is not cached, its three
fits also consume the live budget, allowing at most ten new configurations.
Partial/time-out fit attempts count from the durable fit-start journal; a
three-fold reservation is not falsely reported as three completed fits.

## Evaluation and predefined decision rule

Only optimization.json and its existing train/validation features are read.
The sealed manifest, test audio and microphone recording are never loaded.
The original train/validation/test membership files are never modified.
Preparation makes new checksummed, memory-mapped feature arrays; all iterative
evaluation loads **training arrays only**. No features are recomputed from audio.

Three outer speaker-disjoint development folds cover the 8,400 training clips
exactly once. A seeded speaker-disjoint inner calibration set, about 20% of the
remaining speaker groups, chooses each fold's threshold. Scalers and classifiers
fit only inner-fit speakers; outer held-out speakers never select thresholds.
Both classes must be present in all fit, calibration and evaluation partitions.
The fold memberships, seed 42 and selection objective are persisted before
any benchmark/candidate fit. The original 1,800-example validation partition
is not used for iterative configuration selection.

For each configuration, record clip balanced accuracy, FPR/FNR, synthetic F1,
fold metrics, speaker metrics, fold/speaker variability and actual fitting time.
Some speaker slices may have one class; their balanced accuracy is null and
excluded from the both-class macro average. Pooling uses each fold's own
inner-calibration threshold; the reported pooled 0.5 threshold is applied to
binary decisions, not a common model threshold.

The fixed objective is:

```
0.60 * mean(fold balanced accuracy)
+ 0.30 * mean(both-class speaker balanced accuracy)
+ 0.10 * min(pooled genuine recall, pooled synthetic recall)
- 0.10 * standard deviation(fold balanced accuracy)
```

Accept only an objective gain of at least 0.002. Additionally, compared with
the development baseline, reject mean fold balanced-accuracy regressions greater
than 0.005 or FPR/FNR increases greater than 0.02. These are fixed before
optimization. Stop after two **complete three-role rounds** without an accepted
meaningful gain; an incomplete budget-limited round does not count as a plateau
round. Failed/duplicate proposals consume their API attempts but never initiate
new fits. Previously measured identical-protocol configurations are loaded into
the tried set/history and reused rather than refitted.

If development selects a new best configuration and budget remains, fit it once
on all 8,400 original training clips and compare it once with the existing
candidate on the original validation partition. Its threshold is selected on
that validation set at this final gate. A larger development objective does not
guarantee a gain over 92.92% validation balanced accuracy. No candidate deploys
automatically. These adaptively reused folds and the previously used validation
partition are development evidence, not independent test/generalization proof.

## Agent reasoning and proposal allowlist

- Feature Researcher receives family contributions and out-of-fold genuine
  false-positive associations. It proposes a feature ablation with fixed C=1
  logistic settings, making the hypothesis testable.
- Classifier Researcher compares bounded logistic C/class weights, shallow
  50/100-tree random forests or small histogram boosting on consistent features.
- Robustness Researcher receives anonymous held-out speaker errors, FPR/FNR and
  fold variability, and proposes changes reducing recording-condition shortcuts.

Each response must contain one short hypothesis and one structured proposal.
No additional fields, code, seed change, new feature extraction, dataset or
neural-network proposal is executable. Logistic C is bounded [0.01,100]; class
weight is null/balanced. Forest depth is 3/6, leaves at least three samples and
one worker. Histogram boosting has at most 100 iterations, 15 leaves, depth
three and early stopping disabled to keep conditions deterministic. All numeric
libraries run with one thread during fits. Existing per-clip duration weighting
is retained. Scikit-learn's balanced class weighting uses window-label counts.

Round feedback includes genuine measured development outcomes, best objective,
false-positive feature associations, anonymous worst speaker slices and tried
configuration fingerprints. Earlier Gary Stafford history is labeled as a
different dataset; prior InTheWild validation experiments use a different
protocol and are not directly comparable. Full history remains local; a bounded
recent summary plus prior overview is sent. No speaker names, filenames, audio,
microphone results or sealed-test labels leave the computer.

## Cost accounting

The [Luminal rate card](https://dash.luminal.com/pricing), supplied by the user
on October 11, 2026, records $0.1425/M input tokens, $0.0029/M cached input tokens
and $0.5700/M output tokens. Reasoning is included in output and never charged a
second time in estimates. Cached input is read from provider usage metadata.

Before dispatch, reserve an uncached input estimate of twice the encoded
message bytes plus 2,048 framing tokens and the complete requested output cap.
Reject prompts over 24 KB and any reservation exceeding the remaining $5 cap.
When valid usage arrives, replace the reservation with the rate-card estimate,
including cache-hit discounts. Unknown usage retains the conservative reserve;
out-of-bound/incomplete response usage stops the run before further spending.
This is token/rate-card accounting, not a provider invoice or balance query.

## Evidence and frontend compatibility

`research-history.jsonl` is the full parent-run ledger: real rounds, agent role,
hypothesis, configuration, request/response provenance, token usage, cost basis,
actual fit counts, measured outcomes, previous-best comparison, decisions,
failure reasons and stopping conditions. Each call's exact aggregate prompt is
saved before dispatch. New checksummed cached fold models and any final research
checkpoint are exclusively created. They do not overwrite baseline/candidate
models or become enabled in the inference loader.

`frontend-events.jsonl` conforms to the original v1 event schema **without
changing that schema**. Since it limits rounds to three and fits to twelve,
each parent round is projected as a separately named one-round child run with
local counts: at most three calls/nine fold fits. Final validation is a separate
child run. Fold metrics use actual fold predictions and thresholds. Unrepresentable
new configurations have null legacy proposal fields; the full configuration and
hypothesis remain in the parent research ledger. Counts are never clipped or
misrepresented as parent totals. No frontend code or default event file changes.
The frontend teammate can consume the separate files at a later integration
checkpoint; this preparation does not claim the existing UI already displays
the new hypothesis/cost fields.

## Explicit stages

From the repository root, with no cloud execution:

```sh
.venv/bin/python -m scripts.optimize_inthewild --research --run-dir runs/NEW-UNIQUE-RUN
```

With separate approval for up to three local fold benchmark fits:

```sh
.venv/bin/python -m scripts.optimize_inthewild --research --benchmark --run-dir runs/NEW-UNIQUE-RUN
```

After reviewing the saved preflight and receiving explicit paid-execution
approval, use its exact command:

```sh
.venv/bin/python -m scripts.optimize_inthewild --research --live --run-dir runs/NEW-UNIQUE-RUN
```

An exclusive live-start marker prevents automatic reruns. The preflight hashes
source, checkpoints, existing results, arrays and development folds; changed
inputs block execution. There is no launch during preparation, automatic paid
retry, automatic test evaluation or automatic deployment.

Tests: `python -m pytest -q tests/test_research.py tests/test_agents.py
tests/test_inthewild_agents.py`. Full tests use the existing isolated missing-model
environment for the legacy UI test; they do not change the app's selected model.
