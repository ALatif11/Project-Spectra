"""Local v4 integrity audit and persistent source-group partitions."""

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf

from scripts.download_dataset import check_inventory, manifest
from src.audio import decode_audio

LABELS = {"real": 0, "fake": 1}
PARTITIONS = ("train", "validation", "test")
FIELDS = ("path", "label", "group", "vendor", "split", "seed", "sha256", "pcm_sha256")
# Inspected all 1,866 filenames at revision fcf5344; c/p2 are processing suffixes.
SOURCE_PATTERN = re.compile(r"(yt|el|hg|hu|lv|po|sp)_(\d{4})(?:_c|_p2)?_part_\d+\.flac")


def source_identity(relative_path: str) -> tuple[int, str, str]:
    path = Path(relative_path)
    if len(path.parts) != 2 or path.parts[0] not in LABELS:
        raise ValueError(f"Expected real/*.flac or fake/*.flac: {relative_path}")
    match = SOURCE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unknown source naming pattern; grouping incomplete: {relative_path}")
    vendor, identifier = match.groups()
    label = LABELS[path.parts[0]]
    if (vendor == "yt") != (label == 0):
        raise ValueError(f"Filename source contradicts folder label: {relative_path}")
    return label, f"{vendor}_{identifier}", vendor


def audit_dataset(root: Path, report_path: Path) -> list[dict]:
    report = check_inventory(root)
    report["invalid"] = []
    report["duplicates"] = []
    rows = []
    seen = {}
    for relative in manifest()["files"]:
        path = root / relative
        if not path.is_file() or relative in report["empty"]:
            continue
        try:
            label, group, vendor = source_identity(relative)
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1 or info.format != "FLAC":
                raise ValueError("Expected 16 kHz mono FLAC")
            if not 2.5 <= info.duration <= 13.01:
                raise ValueError("Expected 2.5–13 second v4 clip")
            payload = path.read_bytes()
            recording = decode_audio(payload)
            if recording.is_silent:
                raise ValueError("Clip is silent")
            digest = hashlib.sha256(recording.samples.tobytes()).hexdigest()
            if digest in seen:
                report["duplicates"].append([seen[digest], relative])
            seen[digest] = relative
            rows.append({"path": relative, "label": label, "group": group, "vendor": vendor,
                         "sha256": hashlib.sha256(payload).hexdigest(), "pcm_sha256": digest})
        except (ValueError, RuntimeError, OSError) as exc:
            report["invalid"].append({"path": relative, "error": str(exc)})
    report["valid_count"] = len(rows)
    report["source"] = {k: v for k, v in manifest().items() if k != "files"}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if any(report[key] for key in ("missing", "unexpected", "empty", "readme_missing", "invalid", "duplicates")):
        raise ValueError(f"Dataset integrity failed; inspect {report_path}. No split was created.")
    return rows


def assign_groups(rows: list[dict], seed: int = 42) -> list[dict]:
    """Optimize only sample counts over seeded candidate whole-group assignments.

    Independently require both classes in each split. No audio/features/metrics
    influence the assignment; proportions are approximate because groups vary.
    """
    rng = np.random.default_rng(seed)
    assignment = {}
    target = np.array([0.70, 0.15, 0.15])
    for label in (0, 1):
        groups = sorted({row["group"] for row in rows if int(row["label"]) == label})
        if len(groups) < 3:
            raise ValueError("Need at least three independent source groups per class.")
        counts = np.array([sum(row["group"] == group for row in rows) for group in groups])
        best_loss, best = float("inf"), None
        for _ in range(4096):
            candidate = rng.choice(3, size=len(groups), p=target)
            if len(set(candidate)) != 3:
                continue
            totals = np.bincount(candidate, weights=counts, minlength=3)
            loss = float(np.square(totals / counts.sum() - target).sum())
            if loss < best_loss:
                best_loss, best = loss, candidate.copy()
        if best is None:
            raise ValueError("Could not find valid grouped partitions.")
        assignment.update({group: PARTITIONS[int(split)] for group, split in zip(groups, best)})
    result = [dict(row, split=assignment[row["group"]], seed=seed) for row in rows]
    validate_splits(result)
    return result


def validate_splits(rows: list[dict]) -> None:
    groups, paths, digests = {}, set(), {}
    seeds = {int(row["seed"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError("Split file must have exactly one seed.")
    for row in rows:
        label, group, vendor = source_identity(row["path"])
        if (int(row["label"]), row["group"], row["vendor"]) != (label, group, vendor):
            raise ValueError("Split labels/source identity disagree with filenames.")
        partition = row["split"]
        if partition not in PARTITIONS or row["path"] in paths:
            raise ValueError("Unknown partition or duplicate clip in split file.")
        paths.add(row["path"])
        if groups.setdefault(group, partition) != partition:
            raise ValueError(f"Source leakage across partitions: {group}")
        for key in ("sha256", "pcm_sha256"):
            digest = row[key]
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Missing or invalid audio digest.")
            if (key, digest) in digests:
                raise ValueError("Duplicate recording in split file.")
            digests[key, digest] = partition
    for partition in PARTITIONS:
        if {int(row["label"]) for row in rows if row["split"] == partition} != {0, 1}:
            raise ValueError(f"Both classes required in {partition}.")


def load_splits(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate_splits(rows)
    if {row["path"] for row in rows} != set(manifest()["files"]):
        raise ValueError("Splits must include every clip in the pinned v4 manifest.")
    return rows


def prepare(root: Path, splits: Path, seed: int = 42) -> list[dict]:
    rows = audit_dataset(root, splits.with_name("preparation_report.json"))
    if splits.exists():
        existing = load_splits(splits)
        current = {row["path"]: row for row in rows}
        if any(int(row["seed"]) != seed or any(row[key] != str(current[row["path"]][key])
               for key in ("label", "group", "vendor", "sha256", "pcm_sha256")) for row in existing):
            raise ValueError("Existing partitions differ from requested seed/data; preserve the original split.")
        return existing
    result = assign_groups(rows, seed)
    with splits.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(result)
    return result
