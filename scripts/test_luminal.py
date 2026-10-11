"""One small Luminal chat request; no experiments, audio, retries, or response logging."""

import argparse
from pathlib import Path
import sys

# Support direct execution and python -m scripts.test_luminal from any directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.luminal_client import LuminalClient, LuminalConfig, LuminalError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check configuration without a paid request.")
    args = parser.parse_args(argv)
    try:
        config = LuminalConfig.from_env()
        if args.dry_run:
            print("Luminal configuration valid; dry run made zero API requests.")
            return 0
        LuminalClient(config).complete(
            [{"role": "user", "content": "Reply with only OK."}],
            max_output_tokens=min(16, config.max_output_tokens),
        )
    except LuminalError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ImportError, OSError):
        print("Luminal test could not start; verify optional agent dependencies and root .env access.", file=sys.stderr)
        return 1
    print("Luminal connectivity succeeded: one chat completion, max_tokens <= 16, no retries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
