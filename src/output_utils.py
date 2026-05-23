from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, List, Mapping, Optional, Tuple


def ensure_output_dirs(root: Path) -> Tuple[Path, Path]:
    figs_dir = root / "figs"
    csv_dir = root / "csv"
    figs_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    return figs_dir, csv_dir


def stable_hash(payload: Mapping[str, Any], digits: int = 10) -> str:
    blob = json.dumps(_to_jsonable(payload), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:digits]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            values: List[str] = []
            for key in header:
                value = row.get(key, "")
                if isinstance(value, float):
                    values.append(f"{value:.8g}")
                else:
                    text = str(value)
                    if "," in text or "\"" in text:
                        text = "\"" + text.replace("\"", "\"\"") + "\""
                    values.append(text)
            handle.write(",".join(values) + "\n")


def save_eval_records(records: Iterable[Mapping[str, Any]], save_path: Path) -> None:
    """Unified sink for per-sample evaluation records."""
    write_csv(save_path, records)


def write_run_meta(
    *,
    output_root: Path,
    script_name: str,
    device: str,
    config: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_name,
        "device": device,
        "git_commit": get_git_commit(output_root),
        "config": _to_jsonable(dict(config)),
    }
    if extra:
        meta["extra"] = _to_jsonable(dict(extra))
    meta_path = output_root / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=True)
    return meta_path


def build_tag(*parts: Any) -> str:
    cleaned: List[str] = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in text)
        cleaned.append(safe)
    return "_".join(cleaned)


def get_git_commit(start_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
