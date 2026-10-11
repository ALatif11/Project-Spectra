"""Repository-root environment loading and fixed research budget bounds."""

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    """Load only the root .env, regardless of working directory; preserve host values."""
    from dotenv import load_dotenv

    load_dotenv(REPOSITORY_ROOT / ".env", override=False)


def bounded_env_int(name: str, maximum: int, default: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(maximum if default is None else default)))
    except ValueError:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.") from None
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}; hard caps cannot be raised.")
    return value
