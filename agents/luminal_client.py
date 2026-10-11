"""Luminal's verified OpenAI-compatible chat API; one attempt, no fallback."""

from dataclasses import dataclass, field
import json
import math
import os
import urllib.parse

from agents.configuration import bounded_env_int, load_environment

DEFAULT_BASE_URL = "https://flash.luminal.cloud/v1"
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"


class LuminalError(RuntimeError):
    """Safe, actionable configuration or provider error; never includes raw responses."""


@dataclass(frozen=True)
class LuminalConfig:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model_id: str = DEFAULT_MODEL_ID
    timeout_s: float = 30.0
    max_output_tokens: int = 1000

    def __post_init__(self):
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.api_key, self.base_url, self.model_id)) or self.api_key == "your_luminal_api_key_here":
            raise LuminalError("Configure LUMINAL_API_KEY in the repository-root .env; verify LUMINAL_BASE_URL and LUMINAL_MODEL_ID.")
        try:
            url = urllib.parse.urlsplit(self.base_url)
            url.port
        except ValueError:
            raise LuminalError("Invalid LUMINAL_BASE_URL; verify provider configuration.") from None
        if (url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment):
            raise LuminalError("Use an HTTPS LUMINAL_BASE_URL without credentials or query parameters.")
        if type(self.timeout_s) not in (int, float) or not 0 < self.timeout_s <= 120 or not math.isfinite(self.timeout_s):
            raise LuminalError("LUMINAL_TIMEOUT_S must be finite, greater than zero, and at most 120 seconds.")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 1000:
            raise LuminalError("MAX_OUTPUT_TOKENS must be an integer between 1 and 1000; the hard cap cannot be raised.")
        if any("\n" in value or "\r" in value for value in (self.api_key, self.model_id)):
            raise LuminalError("Credentials and model ID must not contain newlines.")

    @classmethod
    def from_env(cls):
        load_environment()
        try:
            timeout = float(os.environ.get("LUMINAL_TIMEOUT_S", "30"))
            tokens = bounded_env_int("MAX_OUTPUT_TOKENS", 1000)
        except ValueError:
            raise LuminalError("Use a numeric LUMINAL_TIMEOUT_S and MAX_OUTPUT_TOKENS between 1 and 1000.") from None
        return cls(os.environ.get("LUMINAL_API_KEY", ""),
                   os.environ.get("LUMINAL_BASE_URL", DEFAULT_BASE_URL),
                   os.environ.get("LUMINAL_MODEL_ID", DEFAULT_MODEL_ID), timeout, tokens)


class LuminalClient:
    def __init__(self, config: LuminalConfig):
        self.config = config  # Keep this object picklable for the bounded process runner.

    def complete(self, messages: list[dict], *, max_output_tokens: int | None = None) -> str:
        """Make exactly one chat request; callers must supply no audio or secrets."""
        from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI
        import httpx

        tokens = self.config.max_output_tokens if max_output_tokens is None else max_output_tokens
        if type(tokens) is not int or not 1 <= tokens <= self.config.max_output_tokens:
            raise LuminalError("Requested output tokens must be positive and within MAX_OUTPUT_TOKENS.")
        try:
            # Default SDK retries would silently consume the run's call budget.
            # Refuse redirects to preserve the existing credential transport policy.
            with OpenAI(api_key=self.config.api_key, base_url=self.config.base_url,
                        timeout=self.config.timeout_s, max_retries=0,
                        http_client=httpx.Client(follow_redirects=False)) as client:
                result = client.chat.completions.create(
                    model=self.config.model_id, messages=messages, max_tokens=tokens,
                )
            content = result.choices[0].message.content
            if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > 65536:
                raise LuminalError("Luminal returned empty or oversized chat content; no automatic retry was made.")
            return content
        except APIStatusError as exc:
            hint = {
                400: "Request rejected; verify model and max_tokens support. No parameter fallback was attempted.",
                401: "Check LUMINAL_API_KEY.", 403: "Check provider/model access.",
                404: "Verify LUMINAL_BASE_URL and LUMINAL_MODEL_ID.",
                422: "Request parameters rejected; verify max_tokens support. No parameter fallback was attempted.",
                429: "Provider rate limit reached; retry later.",
            }.get(exc.status_code, "Check provider availability and API configuration.")
            raise LuminalError(f"Luminal HTTP {exc.status_code}. {hint} No automatic retry was made.") from None
        except APITimeoutError:
            raise LuminalError("Luminal request timed out. No automatic retry was made.") from None
        except APIConnectionError:
            raise LuminalError("Luminal connection failed; verify network, TLS, and LUMINAL_BASE_URL. No automatic retry was made.") from None
        except APIError:
            raise LuminalError("Luminal SDK response failed validation. No automatic retry was made.") from None
        except (ValueError, AttributeError, KeyError, IndexError, TypeError, RecursionError):
            raise LuminalError("Luminal response does not match the chat completion format. No automatic retry was made.") from None

    def propose(self, role: str, summary: dict) -> dict:
        """All three research roles share this config and anonymized validation summaries."""
        from agents.proposals import PROPOSAL_SCHEMA, ROLE_INSTRUCTIONS, ProposalError, parse_json, validate_proposal

        if role not in ROLE_INSTRUCTIONS:
            raise LuminalError("Unknown proposal role.")
        content = self.complete([
            {"role": "system", "content": "Propose one bounded experiment as plain JSON only. No code or shell commands. "
             + ROLE_INSTRUCTIONS[role] + " JSON schema: " + json.dumps(PROPOSAL_SCHEMA)},
            {"role": "user", "content": json.dumps(summary, allow_nan=False)},
        ])
        try:
            return validate_proposal(parse_json(content), role)
        except ProposalError:
            raise ProposalError("Provider proposal failed the experiment allowlist; raw response was discarded.") from None
