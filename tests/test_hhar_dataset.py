import shutil
import uuid
from pathlib import Path

import torch

from src.datasets import UEATimeSeriesDataset, ViewConfig


def _write_hhar_split(path: Path) -> None:
    payload = {
        "version": 1,
        "dataset": "HHAR",
        "split": "train",
        "data": torch.randn(4, 3, 32, dtype=torch.float32),
        "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "lengths": torch.tensor([32, 32, 32, 32], dtype=torch.long),
        "domain_ids": torch.tensor([2, 3, 2, 3], dtype=torch.long),
        "class_labels": ["bike", "walk"],
        "domain_labels": ["phone:model:nexus4", "watch:model:gear"],
        "meta": {"window_size": 32, "stride": 16},
    }
    torch.save(payload, path)


def _make_local_tmp_root() -> Path:
    root = Path(__file__).resolve().parents[1] / "outputs_new" / "tmp_pytest_hhar" / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_hhar_loader_reads_pt_and_emits_domain_meta():
    root = _make_local_tmp_root() / "all_datasets"
    hhar = root / "HHAR"
    hhar.mkdir(parents=True, exist_ok=True)
    _write_hhar_split(hhar / "HHAR_TRAIN.pt")
    _write_hhar_split(hhar / "HHAR_TEST.pt")

    try:
        ds = UEATimeSeriesDataset(
            name="HHAR",
            split="train",
            root_dir=root,
            return_freq=False,
            use_cache=False,
        )
        assert len(ds) == 4
        item = ds[1]
        assert tuple(item["x_time"].shape) == (3, 32)
        assert int(item["y"].item()) == 1
        assert int(item["meta"]["domain_id"].item()) == 3
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_hhar_loader_can_build_triview():
    root = _make_local_tmp_root() / "all_datasets"
    hhar = root / "HHAR"
    hhar.mkdir(parents=True, exist_ok=True)
    _write_hhar_split(hhar / "HHAR_TRAIN.pt")
    _write_hhar_split(hhar / "HHAR_TEST.pt")

    try:
        ds = UEATimeSeriesDataset(
            name="HHAR",
            split="test",
            root_dir=root,
            return_freq=True,
            view_config=ViewConfig(n_fft=16, hop_length=4),
            use_cache=False,
        )
        item = ds[0]
        assert "x_freq" in item and "x_tf" in item
        assert item["x_freq"].dim() == 2
        assert item["x_tf"].dim() == 2
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)
