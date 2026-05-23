import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FINAL_RE = re.compile(
    r"(?:best_checkpoint_eval=1|final_eval=1).*?test_loss=([0-9.]+)\s+test_acc=([0-9.]+)\s+test_mf1=([0-9.]+)"
)
ANY_ACC_RE = re.compile(r"test_acc=([0-9.]+)")


TRIAL_PLAN: List[Dict[str, object]] = [
    {
        "name": "all_r2_pre0",
        "args": [
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--pretrain-epochs",
            "0",
            "--encoder-lr",
            "1e-4",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "1e-3",
            "--label-smoothing",
            "0.05",
            "--fuse-dropout",
            "0.2",
            "--head-dropout",
            "0.3",
            "--gate-dropout",
            "0.1",
        ],
    },
    {
        "name": "all_r2_pre5",
        "args": [
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--pretrain-epochs",
            "5",
            "--encoder-lr",
            "8e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "8e-4",
            "--label-smoothing",
            "0.05",
            "--fuse-dropout",
            "0.2",
            "--head-dropout",
            "0.3",
            "--gate-dropout",
            "0.1",
        ],
    },
    {
        "name": "all_r2_pre5_cosine",
        "args": [
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--pretrain-epochs",
            "5",
            "--encoder-lr",
            "8e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "8e-4",
            "--label-smoothing",
            "0.0",
            "--fuse-dropout",
            "0.2",
            "--head-dropout",
            "0.2",
            "--gate-dropout",
            "0.0",
            "--use-cosine",
            "--cosine-t0",
            "12",
            "--cosine-t-mult",
            "1",
            "--cosine-eta-min",
            "1e-6",
        ],
    },
    {
        "name": "all_r3_pre5_cosine",
        "args": [
            "--backbone",
            "all",
            "--res-blocks",
            "3",
            "--pretrain-epochs",
            "5",
            "--encoder-lr",
            "6e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "8e-4",
            "--label-smoothing",
            "0.0",
            "--fuse-dropout",
            "0.2",
            "--head-dropout",
            "0.2",
            "--gate-dropout",
            "0.0",
            "--use-cosine",
            "--cosine-t0",
            "12",
            "--cosine-t-mult",
            "1",
            "--cosine-eta-min",
            "1e-6",
        ],
    },
    {
        "name": "incra_r2_pre5",
        "args": [
            "--backbone",
            "inception_resattn",
            "--res-blocks",
            "2",
            "--pretrain-epochs",
            "5",
            "--encoder-lr",
            "1e-4",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "8e-4",
            "--label-smoothing",
            "0.0",
            "--fuse-dropout",
            "0.2",
            "--head-dropout",
            "0.2",
            "--gate-dropout",
            "0.0",
            "--use-cosine",
            "--cosine-t0",
            "12",
            "--cosine-t-mult",
            "1",
            "--cosine-eta-min",
            "1e-6",
        ],
    },
    {
        "name": "all_r2_pre5_mixup",
        "args": [
            "--backbone",
            "all",
            "--res-blocks",
            "2",
            "--pretrain-epochs",
            "5",
            "--encoder-lr",
            "8e-5",
            "--head-lr",
            "2e-4",
            "--weight-decay",
            "8e-4",
            "--label-smoothing",
            "0.0",
            "--mixup-alpha",
            "0.2",
            "--mixup-prob",
            "0.5",
            "--fuse-dropout",
            "0.2",
            "--head-dropout",
            "0.2",
            "--gate-dropout",
            "0.0",
            "--use-cosine",
            "--cosine-t0",
            "12",
            "--cosine-t-mult",
            "1",
            "--cosine-eta-min",
            "1e-6",
        ],
    },
]


def _parse_seeds(raw: str) -> List[int]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--seeds cannot be empty.")
    seeds = [int(t) for t in tokens]
    uniq = []
    for s in seeds:
        if s not in uniq:
            uniq.append(s)
    return uniq


def _build_base_cmd(args: argparse.Namespace, run_name: str, seed: int) -> List[str]:
    return [
        args.python,
        str(args.train_script),
        "--dataset",
        "SelfRegulationSCP2",
        "--dataset-profile",
        "none",
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
        "--seed",
        str(seed),
        "--deterministic",
        "--save-top-k",
        "0",
        "--save-dir",
        str(args.save_dir),
        "--run-name",
        run_name,
        "--supervised-views",
        "triview",
        "--triview-fusion",
        "gated",
        "--use-temporal-attn",
        "--use-shared-qk-attn",
        "--shared-qk-heads",
        "4",
        "--no-freeze-encoder",
        "--normalize-mode",
        "none",
        "--val-split",
        "0.0",
        "--eval-test-each-epoch",
        "--class-weight-mode",
        "none",
        "--train-sampler",
        "none",
        "--n-fft",
        "256",
        "--hop-length",
        "64",
        "--no-pin-memory",
    ]


def _run_trial(cmd: List[str], cwd: Path) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float], str, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    output = proc.stdout or ""

    peak_acc: Optional[float] = None
    for match in ANY_ACC_RE.finditer(output):
        value = float(match.group(1))
        peak_acc = value if peak_acc is None else max(peak_acc, value)

    final_loss: Optional[float] = None
    final_acc: Optional[float] = None
    final_mf1: Optional[float] = None
    last_match = None
    for match in FINAL_RE.finditer(output):
        last_match = match
    if last_match is not None:
        final_loss = float(last_match.group(1))
        final_acc = float(last_match.group(2))
        final_mf1 = float(last_match.group(3))

    return proc.returncode, peak_acc, final_loss, final_acc, final_mf1, output, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Targeted SCP2 tuning. Stops once target accuracy is reached."
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=Path("src/train_uea.py"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="7,11,13,17,19,23,29,31,37,41")
    parser.add_argument("--target-acc", type=float, default=0.65)
    parser.add_argument(
        "--target-metric",
        type=str,
        default="peak_test_acc",
        choices=["peak_test_acc", "final_test_acc"],
    )
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run full plan x seeds.")
    parser.add_argument("--run-prefix", type=str, default="scp2_target65")
    parser.add_argument("--save-dir", type=Path, default=Path("time-main/checkpoints"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs_new/csv/scp2_target65_search.csv"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_script = args.train_script
    if not train_script.is_absolute():
        train_script = project_root / train_script
    if not train_script.exists():
        raise FileNotFoundError(f"train script not found: {train_script}")
    args.train_script = train_script

    seeds = _parse_seeds(args.seeds)
    queue: List[Tuple[str, int, List[str]]] = []
    for trial in TRIAL_PLAN:
        trial_name = str(trial["name"])
        trial_args = list(trial["args"])  # type: ignore[arg-type]
        for seed in seeds:
            queue.append((trial_name, seed, trial_args))
    if args.max_runs > 0:
        queue = queue[: args.max_runs]
    if not queue:
        print("no_trials_to_run=1")
        return 1

    summary_rows: List[Dict[str, object]] = []
    best_score = -1.0
    best_row: Optional[Dict[str, object]] = None

    for i, (trial_name, seed, trial_args) in enumerate(queue, start=1):
        run_name = f"{args.run_prefix}_{i:03d}_{trial_name}_s{seed}"
        cmd = _build_base_cmd(args, run_name, seed) + trial_args
        cmd_display = " ".join(cmd)

        print(f"[{i}/{len(queue)}] trial={trial_name} seed={seed}")
        print(f"cmd={cmd_display}")

        code, peak_acc, final_loss, final_acc, final_mf1, output, elapsed = _run_trial(cmd, cwd=project_root)
        score = peak_acc if args.target_metric == "peak_test_acc" else final_acc

        if code != 0:
            print(f"trial_status=fail exit_code={code} elapsed_sec={elapsed:.1f}")
            tail = "\n".join(output.splitlines()[-20:])
            print("trial_log_tail_start")
            print(tail)
            print("trial_log_tail_end")
        else:
            print(
                f"trial_status=ok elapsed_sec={elapsed:.1f} "
                f"peak_test_acc={peak_acc if peak_acc is not None else float('nan'):.4f} "
                f"final_test_acc={final_acc if final_acc is not None else float('nan'):.4f} "
                f"final_test_mf1={final_mf1 if final_mf1 is not None else float('nan'):.4f}"
            )

        row = {
            "trial_idx": i,
            "trial_name": trial_name,
            "seed": seed,
            "run_name": run_name,
            "exit_code": code,
            "elapsed_sec": round(elapsed, 2),
            "peak_test_acc": "" if peak_acc is None else peak_acc,
            "final_test_loss": "" if final_loss is None else final_loss,
            "final_test_acc": "" if final_acc is None else final_acc,
            "final_test_mf1": "" if final_mf1 is None else final_mf1,
            "score_used": "" if score is None else score,
            "cmd": cmd_display,
        }
        summary_rows.append(row)

        if code == 0 and score is not None and score > best_score:
            best_score = score
            best_row = row

        if code == 0 and score is not None and score >= args.target_acc:
            print(
                f"target_reached=1 target_acc={args.target_acc:.4f} "
                f"target_metric={args.target_metric} achieved={score:.4f} run_name={run_name}"
            )
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
        f"peak_test_acc={best_row['peak_test_acc']} final_test_acc={best_row['final_test_acc']} "
        f"final_test_mf1={best_row['final_test_mf1']}"
    )
    if float(best_score) < args.target_acc:
        print(
            f"target_reached=0 target_acc={args.target_acc:.4f} "
            f"target_metric={args.target_metric} best_score={best_score:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
