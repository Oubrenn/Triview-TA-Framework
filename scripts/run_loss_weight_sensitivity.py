import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = [
    "UWaveGestureLibrary",
    "JapaneseVowels",
    "SpokenArabicDigits",
    "Handwriting",
    "FaceDetection",
    "Heartbeat",
    "HHAR",
]

SETTINGS = [
    {
        "setting": "Default (1,1,1,1)",
        "label": "default",
        "tex": r"Default $(1,1,1,1)$",
        "lambda_md": 1.0,
        "lambda_ta": 1.0,
        "lambda_aux": 1.0,
        "lambda_attn": 1.0,
    },
    {
        "setting": "lambda_md=0.5",
        "label": "md_half",
        "tex": r"$\lambda_{md}=0.5$",
        "lambda_md": 0.5,
        "lambda_ta": 1.0,
        "lambda_aux": 1.0,
        "lambda_attn": 1.0,
    },
    {
        "setting": "lambda_md=2.0",
        "label": "md_x2",
        "tex": r"$\lambda_{md}=2.0$",
        "lambda_md": 2.0,
        "lambda_ta": 1.0,
        "lambda_aux": 1.0,
        "lambda_attn": 1.0,
    },
    {
        "setting": "lambda_ta=0.5",
        "label": "ta_half",
        "tex": r"$\lambda_{ta}=0.5$",
        "lambda_md": 1.0,
        "lambda_ta": 0.5,
        "lambda_aux": 1.0,
        "lambda_attn": 1.0,
    },
    {
        "setting": "lambda_ta=2.0",
        "label": "ta_x2",
        "tex": r"$\lambda_{ta}=2.0$",
        "lambda_md": 1.0,
        "lambda_ta": 2.0,
        "lambda_aux": 1.0,
        "lambda_attn": 1.0,
    },
    {
        "setting": "lambda_aux=0.5",
        "label": "aux_half",
        "tex": r"$\lambda_{aux}=0.5$",
        "lambda_md": 1.0,
        "lambda_ta": 1.0,
        "lambda_aux": 0.5,
        "lambda_attn": 1.0,
    },
    {
        "setting": "lambda_aux=2.0",
        "label": "aux_x2",
        "tex": r"$\lambda_{aux}=2.0$",
        "lambda_md": 1.0,
        "lambda_ta": 1.0,
        "lambda_aux": 2.0,
        "lambda_attn": 1.0,
    },
    {
        "setting": "lambda_attn=0",
        "label": "attn_zero",
        "tex": r"$\lambda_{attn}=0$",
        "lambda_md": 1.0,
        "lambda_ta": 1.0,
        "lambda_aux": 1.0,
        "lambda_attn": 0.0,
    },
]

_VAL_RE = re.compile(r"_ep(?P<epoch>\d+)_val_(?P<metric>acc|mf1|loss)=(?P<value>[0-9.]+)\.pt$")


def _split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(cmd: List[str], cwd: Path, log_path: Path) -> int:
    _ensure_dir(log_path.parent)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        log.write(f"\nelapsed_s={time.perf_counter() - start:.3f}\n")
    return int(proc.returncode)


def _score_checkpoint(path: Path) -> tuple:
    match = _VAL_RE.search(path.name)
    if not match:
        return (float("-inf"), 0, path.stat().st_mtime)
    value = float(match.group("value"))
    epoch = int(match.group("epoch"))
    metric = match.group("metric")
    if metric == "loss":
        value = -value
    return (value, epoch, path.stat().st_mtime)


def _find_checkpoint(save_dir: Path, run_name: str) -> Optional[Path]:
    candidates = list(save_dir.glob(f"{run_name}_ep*_val_*.pt"))
    if not candidates:
        return None
    return sorted(candidates, key=_score_checkpoint, reverse=True)[0]


def _latest_csv(csv_dir: Path, suffix: str) -> Optional[Path]:
    candidates = list(csv_dir.glob(f"*{suffix}"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _fmt_pct(value: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{100.0 * value:.2f}"


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _setting_by_label(label: str) -> Dict[str, object]:
    for item in SETTINGS:
        if item["label"] == label:
            return item
    raise KeyError(label)


def _metrics_from_summary(summary_csv: Path, dataset: str) -> List[Dict[str, object]]:
    rows = _read_csv(summary_csv)
    by_label: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("checkpoint", "")), []).append(row)

    out = []
    for setting in SETTINGS:
        label = str(setting["label"])
        label_rows = by_label.get(label, [])
        clean_rows = [r for r in label_rows if str(r.get("transform", "")) == "clean"]
        pert_rows = [r for r in label_rows if str(r.get("transform", "")) != "clean"]
        if not clean_rows or not pert_rows:
            clean = avg = worst = drop = float("nan")
        else:
            clean = float(clean_rows[0]["acc"])
            vals = [float(r["acc"]) for r in pert_rows]
            avg = sum(vals) / len(vals)
            worst = min(vals)
            drop = clean - worst
        out.append(
            {
                "dataset": dataset,
                "setting": setting["setting"],
                "label": label,
                "lambda_md": setting["lambda_md"],
                "lambda_ta": setting["lambda_ta"],
                "lambda_aux": setting["lambda_aux"],
                "lambda_attn": setting["lambda_attn"],
                "clean_acc": clean,
                "worst_acc": worst,
                "avg_acc": avg,
                "drop": drop,
                "summary_csv": str(summary_csv),
            }
        )
    return out


def _write_tables(output_dir: Path, per_dataset_rows: List[Dict[str, object]]) -> None:
    tables_dir = output_dir / "tables"
    agg_rows = []
    for setting in SETTINGS:
        label = str(setting["label"])
        rows = [r for r in per_dataset_rows if r["label"] == label]
        agg_rows.append(
            {
                "setting": setting["setting"],
                "label": label,
                "lambda_md": setting["lambda_md"],
                "lambda_ta": setting["lambda_ta"],
                "lambda_aux": setting["lambda_aux"],
                "lambda_attn": setting["lambda_attn"],
                "n_datasets": len([r for r in rows if not math.isnan(float(r["clean_acc"]))]),
                "clean_acc": _mean(float(r["clean_acc"]) for r in rows),
                "worst_acc": _mean(float(r["worst_acc"]) for r in rows),
                "avg_acc": _mean(float(r["avg_acc"]) for r in rows),
                "drop": _mean(float(r["drop"]) for r in rows),
            }
        )

    per_fields = [
        "dataset",
        "setting",
        "label",
        "lambda_md",
        "lambda_ta",
        "lambda_aux",
        "lambda_attn",
        "clean_acc",
        "worst_acc",
        "avg_acc",
        "drop",
        "summary_csv",
    ]
    agg_fields = [
        "setting",
        "label",
        "lambda_md",
        "lambda_ta",
        "lambda_aux",
        "lambda_attn",
        "n_datasets",
        "clean_acc",
        "worst_acc",
        "avg_acc",
        "drop",
    ]
    _write_csv(tables_dir / "loss_weight_sensitivity_per_dataset.csv", per_dataset_rows, per_fields)
    _write_csv(tables_dir / "loss_weight_sensitivity_seven_dataset.csv", agg_rows, agg_fields)

    rows_tex = []
    for row in agg_rows:
        setting = _setting_by_label(str(row["label"]))
        rows_tex.append(
            " & ".join(
                [
                    str(setting["tex"]),
                    _fmt_pct(float(row["clean_acc"])),
                    _fmt_pct(float(row["worst_acc"])),
                    _fmt_pct(float(row["avg_acc"])),
                    _fmt_pct(float(row["drop"])),
                ]
            )
            + r" \\"
        )
    rows_path = tables_dir / "loss_weight_sensitivity_rows.tex"
    _ensure_dir(rows_path.parent)
    rows_path.write_text("\n".join(rows_tex) + "\n", encoding="utf-8")

    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{One-factor-at-a-time sensitivity to loss weights on the seven-dataset benchmark.}",
        r"\label{tab:loss_weight_sensitivity}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Setting & Clean Acc & Worst Acc & Avg. Acc & Drop \\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (tables_dir / "loss_weight_sensitivity_table.tex").write_text("\n".join(table), encoding="utf-8")

    md_lines = [
        "| Setting | Clean Acc | Worst Acc | Avg. Acc | Drop | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in agg_rows:
        md_lines.append(
            "| {setting} | {clean} | {worst} | {avg} | {drop} | {n} |".format(
                setting=row["setting"],
                clean=_fmt_pct(float(row["clean_acc"])),
                worst=_fmt_pct(float(row["worst_acc"])),
                avg=_fmt_pct(float(row["avg_acc"])),
                drop=_fmt_pct(float(row["drop"])),
                n=row["n_datasets"],
            )
        )
    (tables_dir / "loss_weight_sensitivity_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compact one-factor loss-weight sensitivity on the seven-dataset benchmark."
    )
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_loss_weight_sensitivity")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sweep-device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sweep-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="all")
    parser.add_argument("--supervised-views", type=str, default="triview")
    parser.add_argument("--triview-fusion", type=str, default="gated")
    parser.add_argument("--shared-qk-heads", type=int, default=4)
    parser.add_argument("--shared-qk-dropout", type=float, default=0.0)
    parser.add_argument("--shift-bins", type=str, default="-0.2,-0.1,0,0.1,0.2")
    parser.add_argument("--scale-ratios", type=str, default="0.9,1,1.1")
    parser.add_argument("--color-max-db", type=str, default="0,3,6")
    parser.add_argument("--shift-fill", type=str, default="border")
    parser.add_argument("--skip-train", action="store_true", default=False)
    parser.add_argument("--skip-sweep", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    datasets = _split_csv(args.datasets)
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset.")

    _ensure_dir(args.output_dir)
    status_path = args.output_dir / "status.json"
    train_script = ROOT / "src" / "train_uea.py"
    sweep_script = ROOT / "scripts" / "sweep_transforms.py"
    per_dataset_rows: List[Dict[str, object]] = []
    train_records = []
    sweep_records = []

    for dataset in datasets:
        dataset_ckpts: List[Path] = []
        for setting in SETTINGS:
            label = str(setting["label"])
            run_name = f"lwsens_{dataset}_{label}_seed{args.seed}"
            ckpt_dir = args.output_dir / "checkpoints" / dataset / label
            checkpoint = _find_checkpoint(ckpt_dir, run_name)
            train_log = args.output_dir / "logs" / "train" / dataset / f"{label}.log"
            if args.force or checkpoint is None:
                if args.skip_train:
                    raise FileNotFoundError(f"Missing checkpoint for {dataset}/{label}: {ckpt_dir}")
                train_cmd = [
                    args.python_exec,
                    str(train_script),
                    "--dataset",
                    dataset,
                    "--dataset-profile",
                    "auto",
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--run-name",
                    run_name,
                    "--save-dir",
                    str(ckpt_dir),
                    "--supervised-views",
                    args.supervised_views,
                    "--triview-fusion",
                    args.triview_fusion,
                    "--backbone",
                    args.backbone,
                    "--hidden-dim",
                    str(args.hidden_dim),
                    "--embed-dim",
                    str(args.embed_dim),
                    "--num-heads",
                    str(args.num_heads),
                    "--res-blocks",
                    str(args.res_blocks),
                    "--pretrain-epochs",
                    str(args.pretrain_epochs),
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
                    "--save-top-k",
                    "1",
                    "--use-shared-qk-attn",
                    "--shared-qk-heads",
                    str(args.shared_qk_heads),
                    "--shared-qk-dropout",
                    str(args.shared_qk_dropout),
                    "--lambda-md",
                    str(setting["lambda_md"]),
                    "--lambda-ta",
                    str(setting["lambda_ta"]),
                    "--lambda-shift",
                    str(setting["lambda_aux"]),
                    "--lambda-scale",
                    str(setting["lambda_aux"]),
                    "--lambda-color",
                    str(setting["lambda_aux"]),
                    "--lambda-attn",
                    str(setting["lambda_attn"]),
                ]
                print(f"[train] dataset={dataset} setting={label}")
                rc = _run(train_cmd, ROOT, train_log)
                checkpoint = _find_checkpoint(ckpt_dir, run_name)
                train_records.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "return_code": rc,
                        "checkpoint": str(checkpoint) if checkpoint is not None else "",
                        "log_path": str(train_log),
                    }
                )
                if rc != 0 or checkpoint is None:
                    raise RuntimeError(f"Training failed for {dataset}/{label}; see {train_log}")
            else:
                train_records.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "return_code": 0,
                        "checkpoint": str(checkpoint),
                        "log_path": str(train_log),
                    }
                )
                print(f"[train-skip] dataset={dataset} setting={label} checkpoint={checkpoint}")
            dataset_ckpts.append(checkpoint)

            status_path.write_text(
                json.dumps(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "stage": "training",
                        "dataset": dataset,
                        "setting": label,
                        "train_records": train_records,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        sweep_root = args.output_dir / "sweeps" / dataset
        sweep_log = args.output_dir / "logs" / "sweep" / f"{dataset}.log"
        summary_csv = _latest_csv(sweep_root / "csv", "_summary.csv")
        if args.force or summary_csv is None:
            if args.skip_sweep:
                raise FileNotFoundError(f"Missing sweep summary for {dataset}: {sweep_root}")
            sweep_cmd = [
                args.python_exec,
                str(sweep_script),
                "--checkpoints",
                *[str(path) for path in dataset_ckpts],
                "--labels",
                ",".join(str(s["label"]) for s in SETTINGS),
                "--dataset",
                dataset,
                "--split",
                "test",
                "--severity-source",
                "fixed",
                "--device",
                args.sweep_device,
                "--batch-size",
                str(args.sweep_batch_size),
                "--num-workers",
                str(args.eval_num_workers),
                f"--shift-bins={args.shift_bins}",
                f"--scale-ratios={args.scale_ratios}",
                f"--color-max-db={args.color_max_db}",
                "--color-trials",
                "1",
                "--shift-fill",
                args.shift_fill,
                "--output-root",
                str(sweep_root),
            ]
            print(f"[sweep] dataset={dataset}")
            rc = _run(sweep_cmd, ROOT, sweep_log)
            summary_csv = _latest_csv(sweep_root / "csv", "_summary.csv")
            sweep_records.append(
                {
                    "dataset": dataset,
                    "return_code": rc,
                    "summary_csv": str(summary_csv) if summary_csv is not None else "",
                    "log_path": str(sweep_log),
                }
            )
            if rc != 0 or summary_csv is None:
                raise RuntimeError(f"Sweep failed for {dataset}; see {sweep_log}")
        else:
            sweep_records.append(
                {
                    "dataset": dataset,
                    "return_code": 0,
                    "summary_csv": str(summary_csv),
                    "log_path": str(sweep_log),
                }
            )
            print(f"[sweep-skip] dataset={dataset} summary={summary_csv}")

        per_dataset_rows.extend(_metrics_from_summary(summary_csv, dataset))
        _write_tables(args.output_dir, per_dataset_rows)
        status_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "stage": "swept",
                    "dataset": dataset,
                    "train_records": train_records,
                    "sweep_records": sweep_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    protocol = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "settings": SETTINGS,
        "args": _jsonable(vars(args)),
        "train_records": train_records,
        "sweep_records": sweep_records,
        "metric_definition": {
            "clean_acc": "test clean accuracy from sweep summary",
            "avg_acc": "mean accuracy over all non-clean shift/scale/color sweep points",
            "worst_acc": "minimum accuracy over all non-clean shift/scale/color sweep points",
            "drop": "clean_acc - worst_acc",
            "aggregate": "macro-average over datasets",
        },
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(_jsonable(protocol), indent=2), encoding="utf-8")
    _write_tables(args.output_dir, per_dataset_rows)
    print(f"saved_tables={args.output_dir / 'tables'}")
    print(f"saved_meta={args.output_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
