import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List


DEFAULT_DATASETS = [
    "SpokenArabicDigits",
    "FaceDetection",
    "Handwriting",
    "Heartbeat",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]


def _parse_datasets(raw: str) -> List[str]:
    parts = [token.strip() for token in re.split(r"[\s,;/]+", raw) if token.strip()]
    parsed: List[str] = []
    seen = set()
    for name in parts:
        if name in seen:
            continue
        seen.add(name)
        parsed.append(name)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run src/train_uea.py sequentially for multiple datasets. "
            "Use '--' to forward extra args to the train script."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=" ".join(DEFAULT_DATASETS),
        help="Dataset names separated by space/comma/semicolon/slash.",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="src/train_uea.py",
        help="Path to the training entry script.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch each run.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=False,
        help="Stop immediately when one dataset run fails.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to train script. Put them after '--'.",
    )
    args = parser.parse_args()

    datasets = _parse_datasets(args.datasets)
    if not datasets:
        parser.error("No valid dataset names were parsed from --datasets.")

    project_root = Path(__file__).resolve().parents[1]
    train_script_path = Path(args.train_script)
    if not train_script_path.is_absolute():
        train_script_path = project_root / train_script_path
    train_script_path = train_script_path.resolve()
    if not train_script_path.exists():
        parser.error(f"train script not found: {train_script_path}")

    forwarded = list(args.train_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    print(f"datasets={datasets}")
    print(f"train_script={train_script_path}")
    print(f"python={args.python}")
    if forwarded:
        print(f"forwarded_args={forwarded}")

    results = []
    for idx, dataset in enumerate(datasets, start=1):
        cmd = [args.python, str(train_script_path), "--dataset", dataset, *forwarded]
        cmd_display = " ".join(shlex.quote(part) for part in cmd)
        print(f"\n[{idx}/{len(datasets)}] dataset={dataset}")
        print(f"cmd={cmd_display}")

        t0 = time.time()
        completed = subprocess.run(cmd, cwd=str(project_root))
        elapsed = time.time() - t0
        results.append((dataset, completed.returncode, elapsed))
        print(f"exit_code={completed.returncode} elapsed_sec={elapsed:.1f}")

        if completed.returncode != 0 and args.stop_on_error:
            break

    failed = [name for name, code, _ in results if code != 0]
    print("\nsummary")
    for name, code, elapsed in results:
        status = "ok" if code == 0 else f"fail({code})"
        print(f"- {name}: {status}, {elapsed:.1f}s")

    if failed:
        print(f"\nfailed_datasets={failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
