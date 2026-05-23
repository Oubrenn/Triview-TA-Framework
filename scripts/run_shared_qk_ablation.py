import argparse
import copy
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from thop import profile

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from train_uea import UEAClassifier, UEATriViewClassifier  # noqa: E402


_FINAL_LINE_RE = re.compile(
    r"(best_checkpoint_eval=1|final_eval=1).*?test_loss=([0-9]*\.?[0-9]+)\s+test_acc=([0-9]*\.?[0-9]+)\s+test_mf1=([0-9]*\.?[0-9]+)"
)


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _extract_final_metrics(log_text: str) -> Dict[str, float]:
    matches = _FINAL_LINE_RE.findall(log_text)
    if not matches:
        return {"test_loss": float("nan"), "test_acc": float("nan"), "test_mf1": float("nan")}
    _, loss, acc, mf1 = matches[-1]
    return {
        "test_loss": float(loss),
        "test_acc": float(acc),
        "test_mf1": float(mf1),
    }


def _build_train_command(
    *,
    python_exec: str,
    train_script: Path,
    dataset: str,
    device: str,
    seed: int,
    run_name: str,
    save_dir: Path,
    shared_qk_on: bool,
    args,
) -> List[str]:
    cmd = [
        python_exec,
        str(train_script),
        "--dataset",
        dataset,
        "--device",
        device,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--save-dir",
        str(save_dir),
        "--dataset-profile",
        args.dataset_profile,
        "--supervised-views",
        args.supervised_views,
        "--triview-fusion",
        args.triview_fusion,
        "--backbone",
        args.backbone,
        "--epochs",
        str(args.epochs),
        "--finetune-epochs",
        str(args.finetune_epochs),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--eval-num-workers",
        str(args.eval_num_workers),
        "--val-split",
        str(args.val_split),
        "--hidden-dim",
        str(args.hidden_dim),
        "--embed-dim",
        str(args.embed_dim),
        "--num-heads",
        str(args.num_heads),
        "--res-blocks",
        str(args.res_blocks),
        "--fuse-dropout",
        str(args.fuse_dropout),
        "--head-dropout",
        str(args.head_dropout),
        "--n-fft",
        str(args.n_fft),
        "--hop-length",
        str(args.hop_length),
    ]
    if args.use_temporal_attn:
        cmd.append("--use-temporal-attn")
    if args.use_se:
        cmd.append("--use-se")
        cmd.extend(["--se-reduction", str(args.se_reduction)])
    if shared_qk_on:
        cmd.append("--use-shared-qk-attn")
        cmd.extend(["--shared-qk-heads", str(args.shared_qk_heads)])
        cmd.extend(["--shared-qk-dropout", str(args.shared_qk_dropout)])
    if args.extra_train_args:
        cmd.extend(args.extra_train_args)
    return cmd


def _build_model_for_efficiency(
    *,
    input_dim_time: int,
    input_dim_freq: int,
    input_dim_tf: int,
    num_classes: int,
    shared_qk_on: bool,
    args,
):
    common = dict(
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_heads=args.num_heads,
        res_blocks=args.res_blocks,
        backbone=args.backbone,
        use_temporal_attn=args.use_temporal_attn,
        use_se=args.use_se,
        se_reduction=args.se_reduction,
        use_shared_qk_attn=shared_qk_on,
        shared_qk_heads=args.shared_qk_heads,
        shared_qk_dropout=args.shared_qk_dropout,
        fuse_dropout=args.fuse_dropout,
        head_dropout=args.head_dropout,
    )
    if args.supervised_views == "triview":
        return UEATriViewClassifier(
            input_dim_time=input_dim_time,
            input_dim_freq=input_dim_freq,
            input_dim_tf=input_dim_tf,
            triview_fusion=args.triview_fusion,
            gate_hidden_dim=args.gate_hidden_dim,
            gate_dropout=args.gate_dropout,
            gate_temperature=args.gate_temperature,
            **common,
        )
    return UEAClassifier(
        input_dim=input_dim_time,
        **common,
    )


def _count_params_m(model: torch.nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters()) / 1e6)


def _measure_latency_ms(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    *,
    warmup: int,
    repeat: int,
    device: str,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(*inputs)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(max(1, repeat)):
            _ = model(*inputs)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / max(1, repeat))


def _measure_efficiency(shared_qk_on: bool, args) -> Dict[str, object]:
    view_cfg = ViewConfig(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        tf_flatten=True,
        tf_log1p=True,
    )
    ds = UEATimeSeriesDataset(
        name=args.dataset,
        split="train",
        pad_to_max=True,
        return_freq=(args.supervised_views == "triview"),
        view_config=view_cfg,
    )
    sample = ds[0]
    x_time_single = sample["x_time"].unsqueeze(0)
    x_freq_single = sample["x_freq"].unsqueeze(0) if args.supervised_views == "triview" else None
    x_tf_single = sample["x_tf"].unsqueeze(0) if args.supervised_views == "triview" else None

    batch_size = args.eff_batch_size
    x_time = x_time_single.repeat(batch_size, 1, 1)
    if x_freq_single is not None:
        x_freq = x_freq_single.repeat(batch_size, 1, 1)
    else:
        x_freq = None
    if x_tf_single is not None:
        x_tf = x_tf_single.repeat(batch_size, 1, 1)
    else:
        x_tf = None

    model = _build_model_for_efficiency(
        input_dim_time=int(x_time.shape[1]),
        input_dim_freq=int(x_freq.shape[1]) if x_freq is not None else 1,
        input_dim_tf=int(x_tf.shape[1]) if x_tf is not None else 1,
        num_classes=len(ds.class_labels),
        shared_qk_on=shared_qk_on,
        args=args,
    ).to(args.eff_device)
    model.eval()

    x_time = x_time.to(args.eff_device)
    inputs: Tuple[torch.Tensor, ...]
    if args.supervised_views == "triview":
        x_freq = x_freq.to(args.eff_device)
        x_tf = x_tf.to(args.eff_device)
        inputs = (x_time, x_freq, x_tf)
    else:
        inputs = (x_time,)

    model_for_macs = copy.deepcopy(model).to(args.eff_device).eval()
    macs_batch, _ = profile(model_for_macs, inputs=inputs, verbose=False)
    macs_single_g = float(macs_batch) / max(1, args.eff_batch_size) / 1e9
    macs_batch_g = float(macs_batch) / 1e9
    latency_ms = _measure_latency_ms(
        model,
        inputs,
        warmup=args.eff_warmup,
        repeat=args.eff_repeat,
        device=args.eff_device,
    )
    hardware = "cpu"
    if str(args.eff_device).startswith("cuda") and torch.cuda.is_available():
        hardware = torch.cuda.get_device_name(torch.cuda.current_device())

    return {
        "params_m": _count_params_m(model),
        "macs_g": macs_single_g,
        "macs_g_batch": macs_batch_g,
        "latency_ms_per_batch": latency_ms,
        "latency_ms_per_sample": latency_ms / max(1, args.eff_batch_size),
        "eff_device": args.eff_device,
        "hardware": hardware,
        "eff_batch_size": args.eff_batch_size,
        "eff_warmup": args.eff_warmup,
        "eff_repeat": args.eff_repeat,
        "input_time_shape": tuple(x_time.shape),
        "input_freq_shape": tuple(x_freq.shape) if x_freq is not None else (),
        "input_tf_shape": tuple(x_tf.shape) if x_tf is not None else (),
    }


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Shared-QK ON/OFF ablation with the same seeds across variants and "
            "the same efficiency measurement protocol."
        )
    )
    parser.add_argument("--dataset", type=str, default="UWaveGestureLibrary")
    parser.add_argument("--dataset-profile", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=ROOT / "src" / "train_uea.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_46" / "shared_qk_ablation")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--supervised-views", type=str, default="triview", choices=["time", "triview"])
    parser.add_argument("--triview-fusion", type=str, default="gated", choices=["concat", "gated"])
    parser.add_argument("--backbone", type=str, default="all")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--use-temporal-attn", action="store_true", default=True)
    parser.add_argument("--use-se", action="store_true", default=False)
    parser.add_argument("--se-reduction", type=int, default=16)
    parser.add_argument("--shared-qk-heads", type=int, default=4)
    parser.add_argument("--shared-qk-dropout", type=float, default=0.0)
    parser.add_argument("--fuse-dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--gate-dropout", type=float, default=0.0)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--eff-device", type=str, default="cuda")
    parser.add_argument("--eff-batch-size", type=int, default=64)
    parser.add_argument("--eff-warmup", type=int, default=30)
    parser.add_argument("--eff-repeat", type=int, default=100)
    parser.add_argument("--extra-train-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    seed_list = _parse_int_list(args.seeds)
    if not seed_list:
        raise ValueError("--seeds must contain at least one integer.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.eff_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested --eff-device cuda but CUDA is not available.")

    results_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    efficiency_rows: List[Dict[str, object]] = []

    variants = [
        ("shared_qk_off", False),
        ("shared_qk_on", True),
    ]
    for variant_name, shared_qk_on in variants:
        for seed in seed_list:
            run_name = f"{args.dataset}_{variant_name}_seed{seed}"
            save_dir = args.output_dir / "checkpoints" / variant_name
            cmd = _build_train_command(
                python_exec=args.python_exec,
                train_script=args.train_script,
                dataset=args.dataset,
                device=args.device,
                seed=seed,
                run_name=run_name,
                save_dir=save_dir,
                shared_qk_on=shared_qk_on,
                args=args,
            )
            start = time.perf_counter()
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            elapsed_s = time.perf_counter() - start
            run_log = completed.stdout
            log_path = args.output_dir / "logs" / f"{run_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(run_log, encoding="utf-8")
            metrics = _extract_final_metrics(run_log)
            results_rows.append(
                {
                    "variant": variant_name,
                    "seed": seed,
                    "shared_qk_on": int(shared_qk_on),
                    "return_code": int(completed.returncode),
                    "elapsed_s": float(elapsed_s),
                    "test_loss": metrics["test_loss"],
                    "test_acc": metrics["test_acc"],
                    "test_mf1": metrics["test_mf1"],
                    "log_path": str(log_path),
                }
            )

        variant_rows = [row for row in results_rows if row["variant"] == variant_name and row["return_code"] == 0]
        acc_values = [float(row["test_acc"]) for row in variant_rows if not torch.isnan(torch.tensor(row["test_acc"]))]
        mf1_values = [float(row["test_mf1"]) for row in variant_rows if not torch.isnan(torch.tensor(row["test_mf1"]))]
        summary_rows.append(
            {
                "variant": variant_name,
                "shared_qk_on": int(shared_qk_on),
                "num_runs": len(variant_rows),
                "seed_list": ",".join(str(s) for s in seed_list),
                "test_acc_mean": _safe_mean(acc_values),
                "test_acc_std": _safe_std(acc_values),
                "test_mf1_mean": _safe_mean(mf1_values),
                "test_mf1_std": _safe_std(mf1_values),
            }
        )
        eff = _measure_efficiency(shared_qk_on=shared_qk_on, args=args)
        efficiency_rows.append(
            {
                "variant": variant_name,
                "shared_qk_on": int(shared_qk_on),
                **eff,
            }
        )

    per_seed_path = args.output_dir / "tables" / "shared_qk_ablation_per_seed.csv"
    summary_path = args.output_dir / "tables" / "shared_qk_ablation_summary.csv"
    efficiency_path = args.output_dir / "tables" / "shared_qk_ablation_efficiency.csv"
    protocol_path = args.output_dir / "tables" / "shared_qk_ablation_protocol.json"

    _write_csv(
        per_seed_path,
        results_rows,
        [
            "variant",
            "seed",
            "shared_qk_on",
            "return_code",
            "elapsed_s",
            "test_loss",
            "test_acc",
            "test_mf1",
            "log_path",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        [
            "variant",
            "shared_qk_on",
            "num_runs",
            "seed_list",
            "test_acc_mean",
            "test_acc_std",
            "test_mf1_mean",
            "test_mf1_std",
        ],
    )
    _write_csv(
        efficiency_path,
        efficiency_rows,
        [
            "variant",
            "shared_qk_on",
            "params_m",
            "macs_g",
            "macs_g_batch",
            "latency_ms_per_batch",
            "latency_ms_per_sample",
            "eff_device",
            "hardware",
            "eff_batch_size",
            "eff_warmup",
            "eff_repeat",
            "input_time_shape",
            "input_freq_shape",
            "input_tf_shape",
        ],
    )

    protocol = {
        "same_seeds_across_variants": True,
        "seed_list": seed_list,
        "same_efficiency_protocol_across_variants": True,
        "efficiency_protocol": {
            "device": args.eff_device,
            "batch_size": args.eff_batch_size,
            "warmup": args.eff_warmup,
            "repeat": args.eff_repeat,
            "macs_definition": "single_input (derived as batch_macs / eff_batch_size)",
        },
        "train_command_common": {
            "dataset": args.dataset,
            "dataset_profile": args.dataset_profile,
            "device": args.device,
            "supervised_views": args.supervised_views,
            "backbone": args.backbone,
            "epochs": args.epochs,
            "finetune_epochs": args.finetune_epochs,
            "pretrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size,
            "val_split": args.val_split,
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"saved_per_seed={per_seed_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_efficiency={efficiency_path}")
    print(f"saved_protocol={protocol_path}")


if __name__ == "__main__":
    main()
