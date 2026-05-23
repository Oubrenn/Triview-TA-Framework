import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RESULT_RE = re.compile(
    r"(?:best_checkpoint_eval=1|final_eval=1).*?test_loss=([0-9.]+)\s+test_acc=([0-9.]+)\s+test_mf1=([0-9.]+)"
)


TRIAL_PLAN: List[Dict[str, object]] = [
    {
        "name": "base_all_r2_noimb",
        "args": [
            "--dataset-profile",
            "none",
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--class-weight-mode",
            "none",
            "--train-sampler",
            "none",
            "--label-smoothing",
            "0.0",
            "--checkpoint-metric",
            "val_acc",
            "--no-freeze-encoder",
            "--encoder-lr",
            "1e-4",
            "--head-lr",
            "3e-4",
            "--weight-decay",
            "1e-3",
        ],
    },
    {
        "name": "attn_all_r2_noimb",
        "args": [
            "--dataset-profile",
            "none",
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--class-weight-mode",
            "none",
            "--train-sampler",
            "none",
            "--label-smoothing",
            "0.0",
            "--checkpoint-metric",
            "val_acc",
            "--no-freeze-encoder",
            "--encoder-lr",
            "5e-5",
            "--head-lr",
            "3e-4",
            "--weight-decay",
            "1e-3",
        ],
    },
    {
        "name": "attn_all_r5_noimb",
        "args": [
            "--dataset-profile",
            "none",
            "--backbone",
            "all",
            "--res-blocks",
            "5",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--class-weight-mode",
            "none",
            "--train-sampler",
            "none",
            "--label-smoothing",
            "0.0",
            "--checkpoint-metric",
            "val_acc",
            "--no-freeze-encoder",
            "--encoder-lr",
            "3e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "1e-3",
        ],
    },
    {
        "name": "pre10_attn_r5_noimb",
        "args": [
            "--dataset-profile",
            "none",
            "--backbone",
            "all",
            "--res-blocks",
            "5",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--pretrain-epochs",
            "10",
            "--class-weight-mode",
            "none",
            "--train-sampler",
            "none",
            "--label-smoothing",
            "0.0",
            "--checkpoint-metric",
            "val_acc",
            "--no-freeze-encoder",
            "--encoder-lr",
            "2e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "2e-3",
        ],
    },
    {
        "name": "pre15_attn_r5_noimb",
        "args": [
            "--dataset-profile",
            "none",
            "--backbone",
            "all",
            "--res-blocks",
            "5",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--pretrain-epochs",
            "15",
            "--class-weight-mode",
            "none",
            "--train-sampler",
            "none",
            "--label-smoothing",
            "0.0",
            "--checkpoint-metric",
            "val_acc",
            "--no-freeze-encoder",
            "--encoder-lr",
            "1e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "5e-3",
        ],
    },
    {
        "name": "profile_auto_pre15_attn_r5",
        "args": [
            "--dataset-profile",
            "auto",
            "--backbone",
            "all",
            "--res-blocks",
            "5",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--pretrain-epochs",
            "15",
            "--checkpoint-metric",
            "val_mf1",
            "--no-freeze-encoder",
            "--encoder-lr",
            "1e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "8e-3",
        ],
    },
    {
        "name": "focal_la_attn_r2",
        "args": [
            "--dataset-profile",
            "none",
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--class-weight-mode",
            "balanced",
            "--train-sampler",
            "none",
            "--loss-type",
            "focal",
            "--focal-gamma",
            "2.0",
            "--logit-adjustment",
            "train_prior",
            "--logit-adjust-tau",
            "1.0",
            "--checkpoint-metric",
            "val_mf1",
            "--no-freeze-encoder",
            "--encoder-lr",
            "3e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "5e-3",
            "--val-split",
            "0.3",
        ],
    },
    {
        "name": "focal_la_attn_r2_long",
        "args": [
            "--dataset-profile",
            "auto",
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--loss-type",
            "focal",
            "--focal-gamma",
            "2.0",
            "--logit-adjustment",
            "train_prior",
            "--logit-adjust-tau",
            "1.0",
            "--checkpoint-metric",
            "val_mf1",
            "--no-freeze-encoder",
            "--encoder-lr",
            "2e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "8e-3",
            "--epochs",
            "100",
            "--finetune-epochs",
            "100",
            "--patience",
            "20",
        ],
    },
    {
        "name": "triview_focal_acc_r2",
        "args": [
            "--dataset-profile",
            "none",
            "--supervised-views",
            "triview",
            "--triview-fusion",
            "gated",
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--loss-type",
            "focal",
            "--focal-gamma",
            "1.5",
            "--class-weight-mode",
            "none",
            "--train-sampler",
            "none",
            "--label-smoothing",
            "0.0",
            "--checkpoint-metric",
            "val_acc",
            "--val-split",
            "0.15",
            "--no-freeze-encoder",
            "--encoder-lr",
            "3e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "5e-3",
        ],
    },
    {
        "name": "triview_focal_mf1_r2",
        "args": [
            "--dataset-profile",
            "auto",
            "--supervised-views",
            "triview",
            "--triview-fusion",
            "gated",
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--use-temporal-attn",
            "--use-shared-qk-attn",
            "--shared-qk-heads",
            "4",
            "--loss-type",
            "focal",
            "--focal-gamma",
            "2.0",
            "--logit-adjustment",
            "train_prior",
            "--logit-adjust-tau",
            "1.0",
            "--logit-adjust-on-eval",
            "--checkpoint-metric",
            "val_mf1",
            "--no-freeze-encoder",
            "--encoder-lr",
            "2e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "8e-3",
        ],
    },
]


def _build_base_cmd(args: argparse.Namespace, run_name: str) -> List[str]:
    return [
        args.python,
        str(args.train_script),
        "--dataset",
        "Heartbeat",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--finetune-epochs",
        str(args.finetune_epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--eval-num-workers",
        str(args.eval_num_workers),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--save-top-k",
        "1",
        "--save-dir",
        str(args.save_dir),
        "--run-name",
        run_name,
        "--no-pin-memory",
    ]


def _run_trial(cmd: List[str], cwd: Path) -> Tuple[int, Optional[float], Optional[float], Optional[float], str, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    output = proc.stdout or ""
    last_match = None
    for match in RESULT_RE.finditer(output):
        last_match = match
    if last_match is None:
        return proc.returncode, None, None, None, output, elapsed
    test_loss = float(last_match.group(1))
    test_acc = float(last_match.group(2))
    test_mf1 = float(last_match.group(3))
    return proc.returncode, test_loss, test_acc, test_mf1, output, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iterative Heartbeat tuning. Stops early when target test_acc is reached."
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=Path("src/train_uea.py"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-acc", type=float, default=0.90)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run full built-in plan.")
    parser.add_argument("--run-prefix", type=str, default="heartbeat_target")
    parser.add_argument("--save-dir", type=Path, default=Path("time-main/checkpoints"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs_new/csv/heartbeat_target_search.csv"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_script = args.train_script
    if not train_script.is_absolute():
        train_script = project_root / train_script
    if not train_script.exists():
        raise FileNotFoundError(f"train script not found: {train_script}")
    args.train_script = train_script

    plan = TRIAL_PLAN[:]
    if args.max_runs > 0:
        plan = plan[: args.max_runs]
    if not plan:
        print("no_trials_to_run=1")
        return 1

    summary_rows: List[Dict[str, object]] = []
    best_acc = -1.0
    best_row: Optional[Dict[str, object]] = None

    for i, trial in enumerate(plan, start=1):
        trial_name = str(trial["name"])
        trial_args = list(trial["args"])  # type: ignore[arg-type]
        run_name = f"{args.run_prefix}_{i:02d}_{trial_name}"
        cmd = _build_base_cmd(args, run_name) + trial_args
        cmd_display = " ".join(cmd)
        print(f"[{i}/{len(plan)}] trial={trial_name}")
        print(f"cmd={cmd_display}")

        code, loss, acc, mf1, output, elapsed = _run_trial(cmd, cwd=project_root)
        if code != 0:
            print(f"trial_status=fail exit_code={code} elapsed_sec={elapsed:.1f}")
        else:
            print(
                f"trial_status=ok elapsed_sec={elapsed:.1f} "
                f"test_loss={loss:.4f} test_acc={acc:.4f} test_mf1={mf1:.4f}"
            )

        row = {
            "trial_idx": i,
            "trial_name": trial_name,
            "run_name": run_name,
            "exit_code": code,
            "elapsed_sec": round(elapsed, 2),
            "test_loss": "" if loss is None else loss,
            "test_acc": "" if acc is None else acc,
            "test_mf1": "" if mf1 is None else mf1,
            "cmd": cmd_display,
        }
        summary_rows.append(row)

        if code == 0 and acc is not None and acc > best_acc:
            best_acc = acc
            best_row = row

        # Keep the tail of logs for failed runs in summary for debugging.
        if code != 0:
            tail = "\n".join(output.splitlines()[-20:])
            print("trial_log_tail_start")
            print(tail)
            print("trial_log_tail_end")

        if code == 0 and acc is not None and acc >= args.target_acc:
            print(f"target_reached=1 target_acc={args.target_acc:.4f} achieved_acc={acc:.4f} run_name={run_name}")
            break

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved_summary_csv={args.summary_csv}")

    if best_row is None:
        print("best_result=none")
        return 1
    print(
        f"best_result run_name={best_row['run_name']} "
        f"test_acc={best_row['test_acc']} test_mf1={best_row['test_mf1']} test_loss={best_row['test_loss']}"
    )
    if float(best_row["test_acc"]) < args.target_acc:
        print(f"target_reached=0 target_acc={args.target_acc:.4f} best_acc={float(best_row['test_acc']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
