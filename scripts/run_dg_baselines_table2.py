import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

_FINAL_LINE_RE = re.compile(
    r"(best_checkpoint_eval=1|final_eval=1).*?test_loss=([0-9]*\.?[0-9]+)\s+test_acc=([0-9]*\.?[0-9]+)\s+test_mf1=([0-9]*\.?[0-9]+)"
)


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_command(
    *,
    python_exec: str,
    train_script: Path,
    dataset: str,
    dataset_profile: str,
    method: str,
    seed: int,
    run_name: str,
    save_dir: Path,
    args,
) -> List[str]:
    cmd = [
        python_exec,
        str(train_script),
        "--dataset",
        dataset,
        "--dataset-profile",
        dataset_profile,
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--save-dir",
        str(save_dir),
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
        "--dg-method",
        method,
        "--dg-lambda",
        str(args.dg_lambda),
        "--dg-min-group-size",
        str(args.dg_min_group_size),
    ]
    if args.use_temporal_attn:
        cmd.append("--use-temporal-attn")
    if args.use_shared_qk_attn:
        cmd.append("--use-shared-qk-attn")
        cmd.extend(["--shared-qk-heads", str(args.shared_qk_heads)])
        cmd.extend(["--shared-qk-dropout", str(args.shared_qk_dropout)])
    if method in {"irm", "rex"} and args.dg_train_with_transforms:
        cmd.append("--dg-train-with-transforms")
    if args.extra_train_args:
        cmd.extend(args.extra_train_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ERM/IRM/REx baselines for Table 2 with aligned seeds and training protocol."
    )
    parser.add_argument("--dataset", type=str, default="UWaveGestureLibrary")
    parser.add_argument("--dataset-profile", type=str, default="auto")
    parser.add_argument("--methods", type=str, default="erm,irm,rex")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=ROOT / "src" / "train_uea.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_46" / "dg_baselines_table2")
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
    parser.add_argument("--use-shared-qk-attn", action="store_true", default=True)
    parser.add_argument("--shared-qk-heads", type=int, default=4)
    parser.add_argument("--shared-qk-dropout", type=float, default=0.0)
    parser.add_argument("--fuse-dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--dg-lambda", type=float, default=1.0)
    parser.add_argument("--dg-min-group-size", type=int, default=2)
    parser.add_argument(
        "--dg-train-with-transforms",
        action="store_true",
        default=False,
        help="Force IRM/REx to use transformed synthetic domains instead of dataset-provided domain_id.",
    )
    parser.add_argument("--extra-train-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    methods = _parse_str_list(args.methods)
    seeds = _parse_int_list(args.seeds)
    if not methods:
        raise ValueError("--methods must contain at least one item.")
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for method in methods:
        for seed in seeds:
            run_name = f"{args.dataset}_{method}_seed{seed}"
            save_dir = args.output_dir / "checkpoints" / method
            cmd = _build_command(
                python_exec=args.python_exec,
                train_script=args.train_script,
                dataset=args.dataset,
                dataset_profile=args.dataset_profile,
                method=method,
                seed=seed,
                run_name=run_name,
                save_dir=save_dir,
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
            log_path = args.output_dir / "logs" / f"{run_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(completed.stdout, encoding="utf-8")
            metrics = _extract_final_metrics(completed.stdout)
            per_seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "return_code": int(completed.returncode),
                    "elapsed_s": float(elapsed_s),
                    "test_loss": metrics["test_loss"],
                    "test_acc": metrics["test_acc"],
                    "test_mf1": metrics["test_mf1"],
                    "log_path": str(log_path),
                }
            )

        ok_rows = [row for row in per_seed_rows if row["method"] == method and row["return_code"] == 0]
        acc_vals = [float(row["test_acc"]) for row in ok_rows]
        mf1_vals = [float(row["test_mf1"]) for row in ok_rows]
        summary_rows.append(
            {
                "method": method,
                "num_runs": len(ok_rows),
                "seed_list": ",".join(str(s) for s in seeds),
                "test_acc_mean": _safe_mean(acc_vals),
                "test_acc_std": _safe_std(acc_vals),
                "test_mf1_mean": _safe_mean(mf1_vals),
                "test_mf1_std": _safe_std(mf1_vals),
            }
        )

    per_seed_path = args.output_dir / "tables" / "dg_baselines_per_seed.csv"
    summary_path = args.output_dir / "tables" / "dg_baselines_summary.csv"
    protocol_path = args.output_dir / "tables" / "dg_baselines_protocol.json"
    _write_csv(
        per_seed_path,
        per_seed_rows,
        ["method", "seed", "return_code", "elapsed_s", "test_loss", "test_acc", "test_mf1", "log_path"],
    )
    _write_csv(
        summary_path,
        summary_rows,
        ["method", "num_runs", "seed_list", "test_acc_mean", "test_acc_std", "test_mf1_mean", "test_mf1_std"],
    )
    protocol = {
        "same_seeds_across_methods": True,
        "seed_list": seeds,
        "methods": methods,
        "train_protocol": {
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
            "dg_lambda": args.dg_lambda,
            "dg_min_group_size": args.dg_min_group_size,
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"saved_per_seed={per_seed_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_protocol={protocol_path}")


if __name__ == "__main__":
    main()
