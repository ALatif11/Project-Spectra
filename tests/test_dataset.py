"""Integrity/grouping checks; generated audio is only a technical test fixture."""

import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts import download_dataset as downloader
from src import dataset


def test_authoritative_manifest():
    source = downloader.manifest()
    assert source["repo_id"] == "garystafford/deepfake-audio-detection"
    assert len(source["files"]) == len(set(source["files"])) == 1866
    assert sum(path.startswith("real/") for path in source["files"]) == 933
    assert sum(path.startswith("fake/") for path in source["files"]) == 933
    identities = [dataset.source_identity(path) for path in source["files"]]
    assert len({group for label, group, _ in identities if label == 0}) == 14
    assert len({group for label, group, _ in identities if label == 1}) == 96


@pytest.mark.parametrize("path,label,group,vendor", [
    ("real/yt_0000_part_001.flac", 0, "yt_0000", "yt"),
    ("real/yt_0000_p2_part_167.flac", 0, "yt_0000", "yt"),
    ("fake/el_0001_part_001.flac", 1, "el_0001", "el"),
    ("fake/el_0001_c_part_002.flac", 1, "el_0001", "el"),
])
def test_explicit_folder_mapping_and_processing_suffixes(path, label, group, vendor):
    assert dataset.source_identity(path) == (label, group, vendor)


@pytest.mark.parametrize("path", ["fake/yt_0000_part_001.flac", "real/el_0001_part_001.flac",
                                  "real/unknown.flac", "../real/yt_0000_part_001.flac"])
def test_incomplete_grouping_or_conflicting_label_rejected(path):
    with pytest.raises(ValueError):
        dataset.source_identity(path)


def manifest_rows():
    rows = []
    for path in downloader.manifest()["files"]:
        label, group, vendor = dataset.source_identity(path)
        digest = hashlib.sha256(path.encode()).hexdigest()
        rows.append(dict(path=path, label=label, group=group, vendor=vendor,
                         sha256=digest, pcm_sha256=digest))
    return rows


def test_seeded_splits_no_source_leakage_both_classes_and_approximate_sizes():
    rows = dataset.assign_groups(manifest_rows(), seed=42)
    assert rows == dataset.assign_groups(manifest_rows(), seed=42)
    dataset.validate_splits(rows)
    group_sets = [{row["group"] for row in rows if row["split"] == split} for split in dataset.PARTITIONS]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    for split, fraction in zip(dataset.PARTITIONS, [0.70, 0.15, 0.15]):
        assert abs(sum(row["split"] == split for row in rows) / len(rows) - fraction) < 0.05
        assert {row["label"] for row in rows if row["split"] == split} == {0, 1}


def test_source_leakage_rejected():
    rows = dataset.assign_groups(manifest_rows())
    rows[1]["split"] = next(split for split in dataset.PARTITIONS if split != rows[0]["split"])
    assert rows[0]["group"] == rows[1]["group"]
    with pytest.raises(ValueError, match="leakage"):
        dataset.validate_splits(rows)


def test_downloader_uses_raw_manifest_and_no_parquet(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        for relative in downloader.manifest()["files"] + ["README.md"]:
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"inventory-test-only")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    assert not any(downloader.download(tmp_path).values())
    assert calls == [dict(repo_id="garystafford/deepfake-audio-detection", repo_type="dataset",
                          revision=downloader.manifest()["revision"],
                          allow_patterns=["real/*.flac", "fake/*.flac", "README.md"],
                          local_dir=str(tmp_path), max_workers=4)]
    assert not list(tmp_path.rglob("*.parquet"))
    assert json.loads((tmp_path / "provenance.json").read_text())["license"] == "CC BY 4.0"


def test_inventory_reports_missing_and_empty(tmp_path):
    relative = downloader.manifest()["files"][0]
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.touch()
    report = downloader.check_inventory(tmp_path)
    assert relative in report["empty"]
    assert len(report["missing"]) == 1865
    assert report["readme_missing"]


def test_imports_do_not_download(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Imports must never download data.")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=forbidden))
    importlib.reload(downloader)
    import src.inference
    importlib.reload(src.inference)


def test_downloader_failure_is_actionable_without_echoing_error_secrets(monkeypatch, capsys):
    def failed(root):
        raise OSError("Potentially sensitive network response")

    monkeypatch.setattr(downloader, "download", failed)
    monkeypatch.setattr(sys, "argv", ["download_dataset"])
    assert downloader.main() == 1
    output = capsys.readouterr().err
    assert "Check network/access and rerun" in output
    assert "Potentially sensitive" not in output


def test_audit_reports_corrupt_and_missing_before_splitting(tmp_path):
    path = tmp_path / "real/yt_0000_part_001.flac"
    path.parent.mkdir()
    path.write_bytes(b"corrupt")
    report_path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="integrity failed"):
        dataset.audit_dataset(tmp_path, report_path)
    report = json.loads(report_path.read_text())
    assert len(report["missing"]) == 1865
    assert report["invalid"][0]["path"] == "real/yt_0000_part_001.flac"


@pytest.fixture
def technical_dataset(tmp_path, monkeypatch):
    """Generated tones exercise mechanics, never detector performance on speech."""
    root = tmp_path / "audio"
    files = []
    for folder, prefix, base_frequency in [("real", "yt", 440), ("fake", "el", 2000)]:
        (root / folder).mkdir(parents=True)
        for source in range(3):
            for part in range(2):
                relative = f"{folder}/{prefix}_{source:04d}_part_{part + 1:03d}.flac"
                time = np.arange(40000) / 16000
                waveform = 0.2 * np.sin(2 * np.pi * (base_frequency + source * 20 + part * 7) * time)
                sf.write(root / relative, waveform, 16000, format="FLAC")
                files.append(relative)
    (root / "README.md").write_text("Technical test tones; not a deepfake benchmark.")
    original = downloader.manifest()
    monkeypatch.setattr(downloader, "manifest", lambda: dict(original, files=files))
    monkeypatch.setattr(dataset, "manifest", lambda: dict(original, files=files))
    return root, tmp_path / "splits.csv"


def test_prepare_persists_and_preserves_partitions(technical_dataset):
    root, splits = technical_dataset
    dataset.prepare(root, splits)
    original = splits.read_bytes()
    dataset.prepare(root, splits)
    assert splits.read_bytes() == original
    assert len(dataset.load_splits(splits)) == 12
    with pytest.raises(ValueError, match="Existing partitions"):
        dataset.prepare(root, splits, seed=43)
    assert splits.read_bytes() == original


def test_duplicate_pcm_rejected(technical_dataset):
    root, splits = technical_dataset
    files = sorted((root / "real").glob("*.flac"))
    files[1].write_bytes(files[0].read_bytes())
    with pytest.raises(ValueError, match="integrity failed"):
        dataset.prepare(root, splits)
    assert not splits.exists()
    assert json.loads(splits.with_name("preparation_report.json").read_text())["duplicates"]
