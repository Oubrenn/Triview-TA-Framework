import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collect_cv_evolution import compute_cross_view_evolution_metrics, write_metrics_csv  # noqa: E402
from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from eval_utils import apply_per_sample_channel  # noqa: E402
from preprocessing import build_triview_from_time  # noqa: E402
from train_uea import UEATriViewClassifier, collate_fn, _domain_stratified_split, _stratified_split  # noqa: E402
from transforms import band_shift_time_stft, make_coloring_gains, spectral_coloring  # noqa: E402


def parse_epoch_from_name(path: Path) -> int:
    match = re.search(r"_ep(\d+)_", path.name)
    if match:
        return int(match.group(1))
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0


def load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict: {path}")
    config = checkpoint.get("config")
    state = checkpoint.get("model_state")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(f"Checkpoint missing config/model_state: {path}")
    return config, state


def view_config_from_config(config: Dict[str, object]) -> ViewConfig:
    return ViewConfig(
        n_fft=int(config.get("n_fft", 256)),
        hop_length=int(config.get("hop_length", 64)),
        win_length=config.get("stft_win_length"),
        window_name=str(config.get("stft_window", "hann")),
        center=bool(config.get("stft_center", True)),
        magnitude_power=float(config.get("stft_magnitude_power", 1.0)),
        tf_log1p=bool(config.get("tf_log1p", True)),
        tf_flatten=bool(config.get("tf_flatten", True)),
        normalize_mode=str(config.get("normalize_mode", "per_sample_channel")),
        shift_mode=str(config.get("pretrain_shift_mode", config.get("shift_fill", "border"))),
    )


def build_triview_model(
    config: Dict[str, object],
    input_dim: int,
    input_dim_freq: int,
    input_dim_tf: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    supervised_views = str(config.get("supervised_views", "time")).strip().lower()
    if supervised_views != "triview":
        raise ValueError(f"Cross-view evolution requires supervised_views=triview, got {supervised_views!r}.")
    model = UEATriViewClassifier(
        input_dim_time=input_dim,
        input_dim_freq=input_dim_freq,
        input_dim_tf=input_dim_tf,
        hidden_dim=int(config.get("hidden_dim", 64)),
        embed_dim=int(config.get("embed_dim", 128)),
        num_classes=num_classes,
        num_heads=int(config.get("num_heads", 4)),
        res_blocks=int(config.get("res_blocks", 2)),
        backbone=str(config.get("backbone", "all")),
        use_temporal_attn=bool(config.get("use_temporal_attn", False)),
        use_se=bool(config.get("use_se", False)),
        se_reduction=int(config.get("se_reduction", 16)),
        use_shared_qk_attn=bool(config.get("use_shared_qk_attn", False)),
        shared_qk_heads=int(config.get("shared_qk_heads", 4)),
        shared_qk_dropout=float(config.get("shared_qk_dropout", 0.0)),
        triview_fusion=str(config.get("triview_fusion", "gated")),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 64)),
        gate_dropout=float(config.get("gate_dropout", 0.0)),
        gate_temperature=float(config.get("gate_temperature", 1.0)),
        fuse_dropout=float(config.get("fuse_dropout", 0.0)),
        head_dropout=float(config.get("head_dropout", 0.0)),
    ).to(device)
    return model


def build_dataset_split(config: Dict[str, object], split: str):
    dataset_name = str(config.get("dataset", "UWaveGestureLibrary"))
    view_config = view_config_from_config(config)
    split = split.lower()
    if split == "test":
        dataset = UEATimeSeriesDataset(
            dataset_name,
            split="test",
            pad_to_max=bool(config.get("pad_to_max", True)),
            return_freq=True,
            view_config=view_config,
            normalize=True,
        )
        return dataset, dataset

    train_full = UEATimeSeriesDataset(
        dataset_name,
        split="train",
        pad_to_max=bool(config.get("pad_to_max", True)),
        return_freq=True,
        view_config=view_config,
        normalize=True,
    )
    val_split = float(config.get("val_split", 0.2))
    if split == "train" or val_split <= 0.0:
        return train_full, train_full

    mode = str(config.get("resolved_val_split_mode", config.get("val_split_mode", "label_stratified")))
    if mode == "auto":
        mode = "domain_stratified" if getattr(train_full, "domain_ids", None) is not None else "label_stratified"
    seed = int(config.get("seed", 42))
    if mode == "domain_stratified" and getattr(train_full, "domain_ids", None) is not None:
        train_indices, val_indices = _domain_stratified_split(train_full.labels, train_full.domain_ids, val_split, seed)
    else:
        train_indices, val_indices = _stratified_split(train_full.labels, val_split, seed)
    if split == "val":
        return torch.utils.data.Subset(train_full, val_indices), train_full
    if split == "train_split":
        return torch.utils.data.Subset(train_full, train_indices), train_full
    raise ValueError("--split must be one of val, test, train, train_split.")


def build_perturb_fn(
    config: Dict[str, object],
    *,
    shift_bins: float,
    color_db: float,
    color_bands: int,
    shift_mode: str,
    seed: int,
):
    view_config = view_config_from_config(config)
    preprocess_config = view_config.to_preprocess_config()

    def perturb(x_time: torch.Tensor) -> Dict[str, torch.Tensor]:
        shifted = apply_per_sample_channel(
            x_time,
            lambda s: band_shift_time_stft(
                s,
                shift_bins=shift_bins,
                n_fft=view_config.n_fft,
                hop_length=view_config.hop_length,
                win_length=view_config.win_length,
                window_name=view_config.window_name,
                center=view_config.center,
                shift_mode=shift_mode,
            ),
        )
        num_bins = shifted.shape[-1] // 2 + 1
        generator = torch.Generator(device="cpu").manual_seed(seed)
        gains = make_coloring_gains(
            num_bins=num_bins,
            bands=color_bands,
            max_gain_db=color_db,
            generator=generator,
        )
        x_pert = apply_per_sample_channel(shifted, lambda s: spectral_coloring(s, gains))
        x_freq_list = []
        x_tf_list = []
        for i in range(x_pert.shape[0]):
            views = build_triview_from_time(x_pert[i], preprocess_config)
            x_freq_list.append(views["x_freq"])
            x_tf_list.append(views["x_tf"])
        return {
            "x_time": x_pert,
            "x_freq": torch.stack(x_freq_list, dim=0).to(x_pert.device),
            "x_tf": torch.stack(x_tf_list, dim=0).to(x_pert.device),
        }

    return perturb


def collect_checkpoint_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        for suffix in ("*.pt", "*.pth", "*.ckpt"):
            paths.extend(path for path in ckpt_dir.glob(suffix) if re.search(r"_ep\d+_", path.name))
    for pattern in args.ckpt_glob:
        paths.extend(Path(path) for path in glob.glob(pattern))
    unique = sorted({path.resolve(): path for path in paths}.values(), key=parse_epoch_from_name)
    if not unique:
        raise FileNotFoundError("No checkpoint files matched --ckpt-dir/--ckpt-glob.")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, default="")
    parser.add_argument("--ckpt-glob", type=str, action="append", default=[])
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train", "train_split"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--mixed-shift-bins", type=float, default=1.0)
    parser.add_argument("--mixed-color-db", type=float, default=3.0)
    parser.add_argument("--mixed-color-bands", type=int, default=8)
    parser.add_argument("--mixed-shift-mode", type=str, default="border", choices=["zero", "circular", "border", "reflect"])
    parser.add_argument("--perturb-seed", type=int, default=20260522)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    ckpt_paths = collect_checkpoint_paths(args)

    first_config, _ = load_checkpoint(ckpt_paths[0], device)
    dataset, shape_dataset = build_dataset_split(first_config, args.split)
    probe = shape_dataset[0]
    input_dim = int(probe["x_time"].shape[0]) if probe["x_time"].dim() > 1 else 1
    input_dim_freq = int(probe["x_freq"].shape[0]) if probe["x_freq"].dim() > 1 else 1
    input_dim_tf = int(probe["x_tf"].shape[0]) if probe["x_tf"].dim() > 1 else 1
    num_classes = len(shape_dataset.class_labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    perturb_fn = build_perturb_fn(
        first_config,
        shift_bins=args.mixed_shift_bins,
        color_db=args.mixed_color_db,
        color_bands=args.mixed_color_bands,
        shift_mode=args.mixed_shift_mode,
        seed=args.perturb_seed,
    )

    rows = []
    for ckpt_path in ckpt_paths:
        config, state = load_checkpoint(ckpt_path, device)
        model = build_triview_model(
            config,
            input_dim=input_dim,
            input_dim_freq=input_dim_freq,
            input_dim_tf=input_dim_tf,
            num_classes=num_classes,
            device=device,
        )
        model.load_state_dict(state, strict=True)
        epoch = int(config.get("epoch", parse_epoch_from_name(ckpt_path)))
        row = compute_cross_view_evolution_metrics(
            model=model,
            val_loader=loader,
            device=device,
            method_name=args.method,
            epoch=parse_epoch_from_name(ckpt_path),
            perturb_fn=perturb_fn,
            max_batches=args.max_batches,
        )
        row["checkpoint"] = str(ckpt_path)
        row["checkpoint_epoch"] = epoch
        row["split"] = args.split
        row["mixed_shift_bins"] = args.mixed_shift_bins
        row["mixed_color_db"] = args.mixed_color_db
        rows.append(row)
        print(
            f"method={args.method} epoch={row['epoch']} "
            f"clean_d_mean={row['clean_d_mean']:.6f} cv_drift={row['cv_drift']:.6f} "
            f"ckpt={ckpt_path}"
        )

    write_metrics_csv(Path(args.out_csv), rows)
    print(f"wrote_csv={args.out_csv} rows={len(rows)}")


if __name__ == "__main__":
    main()
