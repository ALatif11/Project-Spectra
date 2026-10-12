"""Write a sanitized, aggregate-only copy of the project's evidence for machines without the local reports.

    python -m ui.export_evidence --output ui/assets/project_evidence.json

Reads only the allowlisted reports in ui.reporting.SOURCES. Emits aggregate metrics, counts, configurations
and research history with relative source paths and SHA-256 digests. Never includes audio, recording file
names, per-recording predictions, prompts, provider responses, reasoning, credentials or absolute paths.
Run manually after new experiments are recorded; the app never runs it on startup.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ui import reporting as rp

DROP_KEYS = {"external_diagnostics", "per_speaker", "provider_response", "messages", "prompt", "reasoning",
             "reasoning_content", "provider", "luminal_provenance", "clips", "predictions", "path", "test_speakers"}
ABSOLUTE = re.compile(r"^(?:/|[A-Za-z]:[\\/]|\\\\)")


def sanitize(value):
    """Drop private keys recursively and reduce any absolute path to its file name."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if k not in DROP_KEYS}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and ABSOLUTE.match(value):
        return Path(value.replace("\\", "/")).name
    return value


def build(root: Path = rp.ROOT) -> dict:
    sections = {"original": rp.original(root), "expansion": rp.expansion(root), "final": rp.final(root),
                "sessions": rp.sessions(root)}
    clean = {}
    for name, sec in sections.items():
        sources = [{"path": s["path"], "sha256": s["sha256"]} for s in sec.get("sources", []) if s.get("sha256")]
        body = sanitize({k: v for k, v in sec.items() if k != "sources"})
        clean[name] = {**body, "sources": sources}
    fin = sections["final"]
    return {"schema_version": rp.SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_identities": {"final_model_sha256": fin.get("model_sha256"), "baseline_sha256": fin.get("baseline_sha256")},
            "note": "Aggregate evidence only. It does not establish which checkpoint is active on any machine.",
            "sections": clean}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=str(rp.EXPORT_PATH.relative_to(rp.ROOT)))
    out = rp.ROOT / p.parse_args().output
    if rp.ROOT / "ui" not in out.resolve().parents:
        p.error("The export must be written under ui/.")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    out.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(rp.ROOT)} with sections: " +
          ", ".join(f"{k}={v['status']}" for k, v in data["sections"].items()))


if __name__ == "__main__":
    main()
