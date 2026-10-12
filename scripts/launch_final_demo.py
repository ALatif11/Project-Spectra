"""Launch the verified frozen candidate, or its preserved baseline rollback."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys

from src.inthewild_final import FREEZE, ROOT
from src.model import file_digest, load_model


def local_path(recorded: str, freeze: dict) -> Path:
    """The freeze record holds absolute paths from the machine that made it; rebase them onto this checkout.

    The SHA-256 check in selected_path still guards the file, so rebasing cannot swap in another checkpoint.
    """
    if Path(recorded).exists():
        return Path(recorded)
    posix = lambda p: PurePosixPath(p.replace("\\", "/"))
    return ROOT / posix(recorded).relative_to(posix(freeze["model_path"]).parents[1])


def selected_path(rollback=False, freeze_path=FREEZE):
    freeze = json.loads(Path(freeze_path).read_text())
    path = local_path(freeze["rollback_path" if rollback else "model_path"], freeze)
    expected = freeze["baseline_sha256" if rollback else "model_sha256"]
    if file_digest(path) != expected:
        raise ValueError("Demo checkpoint changed; preserve artifacts and inspect the freeze record.")
    load_model(path)
    return path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rollback", action="store_true")
    args, streamlit_args = parser.parse_known_args()
    try:
        path = selected_path(args.rollback)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    os.chdir(ROOT)
    os.environ["ECHOPROOF_MODEL_PATH"] = str(path)
    print(f"Demo model: {path.name}", flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                              *streamlit_args])


if __name__ == "__main__":
    main()
