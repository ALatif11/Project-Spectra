"""Hard research ceilings and conservative per-request spending reservations."""

from dataclasses import asdict, dataclass
from decimal import Decimal
import json
import math
import time

from agents.experiment_runner import Limits


@dataclass(frozen=True)
class ResearchPolicy:
    rounds: int = 6
    agents_per_round: int = 3
    llm_calls: int = 60
    experiments: int = 36  # Actual fit attempts, including development folds.
    max_output_tokens: int = 2500
    estimated_cost_usd: float = 5.0
    run_timeout_s: float = 900.0
    train_timeout_s: float = 120.0
    request_timeout_s: float = 60.0
    development_folds: int = 3
    patience: int = 2
    minimum_improvement: float = .002

    def __post_init__(self):
        Limits(self.rounds, self.agents_per_round, self.llm_calls, self.experiments,
               self.train_timeout_s, self.run_timeout_s)
        if self.agents_per_round != 3 or self.development_folds != 3 or self.patience != 2:
            raise ValueError("Use exactly three roles, three development folds and patience of two complete rounds.")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 2500:
            raise ValueError("Research output ceiling is 2500 tokens.")
        for name, maximum in (("estimated_cost_usd", 5), ("request_timeout_s", 120), ("minimum_improvement", .05)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"Invalid {name}; maximum is {maximum}.")


@dataclass(frozen=True)
class Pricing:
    input_per_million: float = .1425
    cached_input_per_million: float = .0029
    output_per_million: float = .57
    source: str = "https://dash.luminal.com/pricing"
    verified_on: str = "2026-10-11 (user-supplied Luminal rate card)"
    model_id: str = "deepseek-ai/DeepSeek-V4.1-Flash"

    def __post_init__(self):
        for value in (self.input_per_million, self.cached_input_per_million, self.output_per_million):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Recorded positive finite Luminal pricing is required.")
        if self.cached_input_per_million > self.input_per_million or not self.source or not self.verified_on:
            raise ValueError("Invalid Luminal rate card.")

    def cost(self, prompt, output, cached=0):
        if any(type(n) is not int or n < 0 for n in (prompt, output, cached)) or cached > prompt:
            raise ValueError("Invalid provider token usage.")
        return ((Decimal(str(self.input_per_million)) * (prompt-cached)
                 + Decimal(str(self.cached_input_per_million)) * cached
                 + Decimal(str(self.output_per_million)) * output) / Decimal(1000000))


class Budget:
    def __init__(self, policy, pricing, *, clock=time.monotonic):
        self.policy, self.pricing, self.clock = policy, pricing, clock
        self.deadline = clock() + policy.run_timeout_s
        self.calls = self.fits = self.fits_completed = 0
        self.cost = Decimal(0)
        self.uncertain_requests = 0

    def remaining(self):
        return max(0.0, self.deadline-self.clock())

    def reserve_call(self, messages):
        if self.remaining() <= 0:
            raise ValueError("run_timeout")
        if self.calls >= self.policy.llm_calls:
            raise ValueError("llm_call_limit")
        size = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if size > 24000:
            raise ValueError("prompt_size_limit")
        # Byte-based, deliberately conservative; no cache discount is assumed.
        # Extra margin covers framing/tokenizer differences without a paid probe.
        reserved_input = 2*size + 2048
        reservation = self.pricing.cost(reserved_input, self.policy.max_output_tokens)
        if self.cost + reservation > Decimal(str(self.policy.estimated_cost_usd)):
            raise ValueError("estimated_cost_limit")
        self.calls += 1
        self.cost += reservation
        return {"reserved_cost_usd": float(reservation), "reserved_prompt_tokens": reserved_input,
                "requested_output_tokens": self.policy.max_output_tokens}

    def settle_call(self, reservation, usage):
        cap = Decimal(str(reservation["reserved_cost_usd"]))
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if (type(prompt) is not int or type(output) is not int or prompt < 0 or output < 0
                or prompt > reservation["reserved_prompt_tokens"] or output > reservation["requested_output_tokens"]):
            self.uncertain_requests += 1
            return {"estimated_cost_usd": float(cap), "cost_basis": "reserved_upper_estimate",
                    "usage_unverified": True}
        cached = usage.get("cached_prompt_tokens", 0)
        if type(cached) is not int or not 0 <= cached <= prompt:
            cached = 0
        actual = self.pricing.cost(prompt, output, cached)
        self.cost += actual-cap
        return {"estimated_cost_usd": float(actual), "cost_basis": "provider_usage_and_recorded_rates",
                "usage_unverified": False}

    def can_fit(self, count, *, reserve_final=1):
        return (self.remaining() > 0 and self.fits + count + reserve_final <= self.policy.experiments)

    def account_fits(self, journal_events):
        self.fits += sum(e["event"] == "fit_started" for e in journal_events)
        self.fits_completed += sum(e["event"] == "fit_completed" for e in journal_events)
        if self.fits > self.policy.experiments:
            raise RuntimeError("Actual fit journal exceeded the research cap.")

    def snapshot(self):
        return {"api_calls": self.calls, "ml_fit_attempts": self.fits, "ml_fits_completed": self.fits_completed,
                "estimated_cost_usd": float(self.cost), "usage_unverified_requests": self.uncertain_requests,
                "remaining_runtime_s": self.remaining(), "limits": asdict(self.policy)}
