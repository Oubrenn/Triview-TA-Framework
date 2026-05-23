import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


TRANSFORM_ORDER: Tuple[str, ...] = ("shift", "scale", "color", "mixed_shift_color")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _preferred_trial(candidates: List[Dict[str, str]]) -> Dict[str, str]:
    mean_row = next((r for r in candidates if str(r.get("trial", "")) == "mean"), None)
    if mean_row is not None:
        return mean_row
    zero_row = next((r for r in candidates if str(r.get("trial", "")) in {"0", "0.0"}), None)
    if zero_row is not None:
        return zero_row
    return candidates[0]


def _canonicalize_rows(rows: List[Dict[str, str]], key_fields: Sequence[str]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        grouped.setdefault(key, []).append(row)
    return [_preferred_trial(grouped[k]) for k in grouped]


def _value_key(row: Dict[str, str], transform_name: str):
    if transform_name == "shift":
        return round(_safe_float(row.get("b_bins", 0.0), 0.0), 6)
    if transform_name == "scale":
        return round(_safe_float(row.get("rho", 1.0), 1.0), 6)
    if transform_name == "color":
        return round(_safe_float(row.get("g_db", 0.0), 0.0), 6)
    if transform_name == "mixed_shift_color":
        return int(round(_safe_float(row.get("severity_id", 0.0), 0.0)))
    return None


def _safe_values_from_teacher(teacher_rows: List[Dict[str, str]], gamma: float) -> Dict[str, Set[object]]:
    safe_values: Dict[str, Set[object]] = {}
    for row in teacher_rows:
        transform_name = str(row.get("transform", "")).strip()
        if not transform_name:
            continue
        agreement = _safe_float(row.get("agreement", 0.0), 0.0)
        if agreement + 1e-12 < gamma:
            continue
        key = _value_key(row, transform_name)
        if key is None:
            continue
        safe_values.setdefault(transform_name, set()).add(key)
    return safe_values


def _clone_safe_values(safe_values: Dict[str, Set[object]]) -> Dict[str, Set[object]]:
    return {k: set(v) for k, v in safe_values.items()}


def _majority_vote(values_by_ref: List[Dict[str, Set[object]]]) -> Dict[str, Set[object]]:
    if not values_by_ref:
        return {}
    threshold = int(math.ceil(len(values_by_ref) / 2.0))
    transforms = sorted({t for item in values_by_ref for t in item.keys()})
    out: Dict[str, Set[object]] = {}
    for transform_name in transforms:
        union_values: Set[object] = set()
        for item in values_by_ref:
            union_values.update(item.get(transform_name, set()))
        kept: Set[object] = set()
        for value in union_values:
            votes = sum(1 for item in values_by_ref if value in item.get(transform_name, set()))
            if votes >= threshold:
                kept.add(value)
        out[transform_name] = kept
    return out


def _strict_intersection(values_by_ref: List[Dict[str, Set[object]]]) -> Dict[str, Set[object]]:
    if not values_by_ref:
        return {}
    transforms = sorted({t for item in values_by_ref for t in item.keys()})
    out: Dict[str, Set[object]] = {}
    for transform_name in transforms:
        current: Optional[Set[object]] = None
        for item in values_by_ref:
            one = set(item.get(transform_name, set()))
            if current is None:
                current = one
            else:
                current = current.intersection(one)
        out[transform_name] = current if current is not None else set()
    return out


def _flatten_safe_set(safe_values: Dict[str, Set[object]]) -> Set[Tuple[str, object]]:
    flattened: Set[Tuple[str, object]] = set()
    for transform_name, values in safe_values.items():
        for value in values:
            flattened.add((transform_name, value))
    return flattened


def _safe_count_by_transform(safe_values: Dict[str, Set[object]]) -> Dict[str, int]:
    out = {}
    for transform_name in TRANSFORM_ORDER:
        if transform_name in safe_values:
            out[f"n_{transform_name}"] = len(safe_values[transform_name])
    # Keep deterministic visibility for any custom transforms.
    for transform_name in sorted(safe_values.keys()):
        key = f"n_{transform_name}"
        if key not in out:
            out[key] = len(safe_values[transform_name])
    return out


def _filter_rows_by_safe(rows: List[Dict[str, str]], safe_values: Dict[str, Set[object]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        transform_name = str(row.get("transform", "")).strip()
        if transform_name == "clean":
            out.append(row)
            continue
        allowed = safe_values.get(transform_name, set())
        if not allowed:
            continue
        key = _value_key(row, transform_name)
        if key in allowed:
            out.append(row)
    return out


def _calc_mixed_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    transformed = [r for r in rows if str(r.get("transform", "")) != "clean"]
    if not transformed:
        return {
            "retained_count": 0.0,
            "mixed_avg_acc": math.nan,
            "mixed_worst_acc": math.nan,
            "mixed_avg_mf1": math.nan,
            "mixed_worst_mf1": math.nan,
        }
    acc_vals = [_safe_float(r.get("acc", 0.0), 0.0) for r in transformed]
    mf1_vals = [_safe_float(r.get("mf1", 0.0), 0.0) for r in transformed]
    return {
        "retained_count": float(len(transformed)),
        "mixed_avg_acc": float(sum(acc_vals) / len(acc_vals)),
        "mixed_worst_acc": float(min(acc_vals)),
        "mixed_avg_mf1": float(sum(mf1_vals) / len(mf1_vals)),
        "mixed_worst_mf1": float(min(mf1_vals)),
    }


def _group_rows_by(rows: List[Dict[str, object]], fields: Sequence[str]) -> Dict[Tuple[object, ...], List[Dict[str, object]]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def _attach_dense_rank(
    rows: List[Dict[str, object]],
    group_fields: Sequence[str],
    method_field: str,
    metric_field: str,
    rank_field: str,
) -> None:
    grouped = _group_rows_by(rows, group_fields)
    for _, members in grouped.items():
        valid = [r for r in members if not math.isnan(float(r.get(metric_field, math.nan)))]
        valid.sort(key=lambda r: (-float(r[metric_field]), str(r.get(method_field, ""))))
        current_rank = 0
        prev_value = None
        for idx, row in enumerate(valid, start=1):
            value = float(row[metric_field])
            if prev_value is None or abs(value - prev_value) > 1e-12:
                current_rank = idx
                prev_value = value
            row[rank_field] = current_rank
        for row in members:
            if rank_field not in row:
                row[rank_field] = ""


def _ranking_string(rows: List[Dict[str, object]], metric_field: str, method_field: str = "method") -> str:
    valid = [r for r in rows if not math.isnan(float(r.get(metric_field, math.nan)))]
    if not valid:
        return ""
    valid.sort(key=lambda r: (-float(r[metric_field]), str(r.get(method_field, ""))))
    return " > ".join(str(r.get(method_field, "")) for r in valid)


def _jaccard(a: Set[Tuple[str, object]], b: Set[Tuple[str, object]]) -> Tuple[float, int, int]:
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union == 0:
        return 1.0, inter, union
    return inter / union, inter, union


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_gamma_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def _default_target_method(methods: List[str]) -> str:
    if "full" in methods:
        return "full"
    return methods[0] if methods else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-model rule sensitivity and gamma sensitivity analysis. "
            "Reads teacher_agreement summaries and sweep summaries, then exports experiment tables."
        )
    )
    parser.add_argument(
        "--teacher-summary-csvs",
        type=str,
        nargs="+",
        required=True,
        help="List of teacher_agreement *_summary.csv files (at least 3 for Ref-1/2/3).",
    )
    parser.add_argument(
        "--ref-labels",
        type=str,
        default="",
        help="Optional labels for references, comma-separated, e.g. Ref-1,Ref-2,Ref-3",
    )
    parser.add_argument(
        "--summary-csvs",
        type=str,
        nargs="+",
        required=True,
        help="One or more sweep *_summary.csv files. Rows will be concatenated then deduplicated.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Optional ordered method list, comma-separated. Default uses all checkpoints in summary rows.",
    )
    parser.add_argument(
        "--target-method",
        type=str,
        default="",
        help="Method used to produce compact experiment tables. Default prefers 'full' if present.",
    )
    parser.add_argument("--gamma-fixed", type=float, default=0.95, help="Gamma used in Experiment 1.")
    parser.add_argument(
        "--gamma-values",
        type=str,
        default="0.90,0.92,0.95,0.97,0.99",
        help="Comma-separated gamma grid for Experiment 2.",
    )
    parser.add_argument(
        "--gamma-rule-ids",
        type=str,
        default="",
        help=(
            "Comma-separated rule ids for Experiment 2. "
            "Supported ids: ref1,ref2,ref3,...,majority,intersection."
        ),
    )
    parser.add_argument("--output-root", type=str, default="outputs_ref_sensitivity")
    args = parser.parse_args()

    teacher_paths = [Path(p) for p in args.teacher_summary_csvs]
    for path in teacher_paths:
        if not path.exists():
            raise FileNotFoundError(f"teacher summary csv not found: {path}")
    summary_paths = [Path(p) for p in args.summary_csvs]
    for path in summary_paths:
        if not path.exists():
            raise FileNotFoundError(f"summary csv not found: {path}")

    ref_labels = _parse_csv_list(args.ref_labels)
    if ref_labels and len(ref_labels) != len(teacher_paths):
        raise ValueError(
            f"--ref-labels count ({len(ref_labels)}) must match --teacher-summary-csvs count ({len(teacher_paths)})."
        )
    if not ref_labels:
        ref_labels = [f"Ref-{idx + 1}" for idx in range(len(teacher_paths))]

    teacher_rows_by_ref: Dict[str, List[Dict[str, str]]] = {}
    for label, path in zip(ref_labels, teacher_paths):
        rows_raw = _read_csv(path)
        rows = _canonicalize_rows(
            rows_raw,
            key_fields=("transform", "severity_id", "rho", "g_db", "b_bins"),
        )
        teacher_rows_by_ref[label] = rows

    summary_rows_raw: List[Dict[str, str]] = []
    for path in summary_paths:
        summary_rows_raw.extend(_read_csv(path))
    summary_rows = _canonicalize_rows(
        summary_rows_raw,
        key_fields=("checkpoint", "transform", "severity_id", "rho", "g_db", "b_bins", "run_id"),
    )

    methods_available = sorted(
        {
            str(row.get("checkpoint", "")).strip()
            for row in summary_rows
            if str(row.get("checkpoint", "")).strip()
        }
    )
    if not methods_available:
        raise ValueError("No methods found in provided summary csv rows.")

    requested_methods = _parse_csv_list(args.methods)
    if requested_methods:
        missing = [m for m in requested_methods if m not in methods_available]
        if missing:
            raise ValueError(f"Requested methods not present in summary rows: {missing}")
        methods = requested_methods
    else:
        methods = methods_available

    target_method = args.target_method.strip() or _default_target_method(methods)
    if target_method not in methods:
        raise ValueError(f"target method '{target_method}' is not in selected methods: {methods}")

    # Rule registry.
    rule_specs: List[Dict[str, str]] = []
    for idx, ref_label in enumerate(ref_labels, start=1):
        rule_specs.append(
            {
                "id": f"ref{idx}",
                "label": f"Single-ref / {ref_label}",
                "type": "single",
                "ref_label": ref_label,
            }
        )
    rule_specs.append({"id": "majority", "label": "Majority consensus", "type": "majority", "ref_label": ""})
    rule_specs.append({"id": "intersection", "label": "Strict intersection", "type": "intersection", "ref_label": ""})
    rule_by_id = {spec["id"]: spec for spec in rule_specs}

    def build_safe_by_rule(gamma: float) -> Dict[str, Dict[str, Set[object]]]:
        safe_by_ref = {
            ref_label: _safe_values_from_teacher(teacher_rows_by_ref[ref_label], gamma=gamma)
            for ref_label in ref_labels
        }
        all_ref_values = [safe_by_ref[label] for label in ref_labels]
        out: Dict[str, Dict[str, Set[object]]] = {}
        for spec in rule_specs:
            if spec["type"] == "single":
                out[spec["id"]] = _clone_safe_values(safe_by_ref[spec["ref_label"]])
            elif spec["type"] == "majority":
                out[spec["id"]] = _majority_vote(all_ref_values)
            elif spec["type"] == "intersection":
                out[spec["id"]] = _strict_intersection(all_ref_values)
            else:
                raise ValueError(f"Unknown rule type: {spec['type']}")
        return out

    def eval_rules(gamma: float, active_rule_ids: Iterable[str]) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, Set[object]]]]:
        safe_by_rule = build_safe_by_rule(gamma=gamma)
        rows_out: List[Dict[str, object]] = []
        for rule_id in active_rule_ids:
            if rule_id not in rule_by_id:
                raise ValueError(f"Unknown rule id: {rule_id}")
            spec = rule_by_id[rule_id]
            safe_values = safe_by_rule[rule_id]
            safe_flat = _flatten_safe_set(safe_values)
            safe_size = len(safe_flat)
            counts = _safe_count_by_transform(safe_values)
            for method in methods:
                method_rows = [r for r in summary_rows if str(r.get("checkpoint", "")).strip() == method]
                filtered = _filter_rows_by_safe(method_rows, safe_values=safe_values)
                metrics = _calc_mixed_metrics(filtered)
                row = {
                    "gamma": gamma,
                    "rule_id": rule_id,
                    "rule_label": spec["label"],
                    "method": method,
                    "safe_set_size": safe_size,
                    **counts,
                    **metrics,
                }
                rows_out.append(row)
        _attach_dense_rank(
            rows_out,
            group_fields=("gamma", "rule_id"),
            method_field="method",
            metric_field="mixed_worst_acc",
            rank_field="rank_worst_acc",
        )
        _attach_dense_rank(
            rows_out,
            group_fields=("gamma", "rule_id"),
            method_field="method",
            metric_field="mixed_worst_mf1",
            rank_field="rank_worst_mf1",
        )
        return rows_out, safe_by_rule

    gamma_fixed = float(args.gamma_fixed)
    exp1_rows, exp1_safe_by_rule = eval_rules(gamma=gamma_fixed, active_rule_ids=[spec["id"] for spec in rule_specs])
    exp1_rule_to_rows = _group_rows_by(exp1_rows, fields=("rule_id",))

    exp1_summary_rows: List[Dict[str, object]] = []
    exp1_target_rows: List[Dict[str, object]] = []
    for spec in rule_specs:
        members = exp1_rule_to_rows.get((spec["id"],), [])
        target = next((r for r in members if str(r.get("method", "")) == target_method), None)
        safe_flat = _flatten_safe_set(exp1_safe_by_rule[spec["id"]])
        row = {
            "gamma": gamma_fixed,
            "rule_id": spec["id"],
            "rule_label": spec["label"],
            "safe_set_size": len(safe_flat),
            **_safe_count_by_transform(exp1_safe_by_rule[spec["id"]]),
            "ranking_worst_acc": _ranking_string(members, metric_field="mixed_worst_acc"),
            "ranking_worst_mf1": _ranking_string(members, metric_field="mixed_worst_mf1"),
        }
        if target is not None:
            row.update(
                {
                    "target_method": target_method,
                    "target_retained_count": target["retained_count"],
                    "target_mixed_avg_acc": target["mixed_avg_acc"],
                    "target_mixed_worst_acc": target["mixed_worst_acc"],
                    "target_mixed_avg_mf1": target["mixed_avg_mf1"],
                    "target_mixed_worst_mf1": target["mixed_worst_mf1"],
                }
            )
            exp1_target_rows.append(
                {
                    "rule_id": spec["id"],
                    "rule_label": spec["label"],
                    "target_method": target_method,
                    "safe_set_size": row["safe_set_size"],
                    "retained_count": target["retained_count"],
                    "mixed_avg_acc": target["mixed_avg_acc"],
                    "mixed_worst_acc": target["mixed_worst_acc"],
                    "mixed_avg_mf1": target["mixed_avg_mf1"],
                    "mixed_worst_mf1": target["mixed_worst_mf1"],
                }
            )
        exp1_summary_rows.append(row)

    exp1_jaccard_rows: List[Dict[str, object]] = []
    for i, lhs in enumerate(rule_specs):
        lhs_flat = _flatten_safe_set(exp1_safe_by_rule[lhs["id"]])
        for j, rhs in enumerate(rule_specs):
            if j <= i:
                continue
            rhs_flat = _flatten_safe_set(exp1_safe_by_rule[rhs["id"]])
            score, inter_size, union_size = _jaccard(lhs_flat, rhs_flat)
            exp1_jaccard_rows.append(
                {
                    "gamma": gamma_fixed,
                    "rule_i": lhs["label"],
                    "rule_j": rhs["label"],
                    "intersection_size": inter_size,
                    "union_size": union_size,
                    "jaccard": score,
                }
            )

    gamma_values = _parse_gamma_list(args.gamma_values)
    if not gamma_values:
        raise ValueError("No gamma values parsed from --gamma-values.")

    if args.gamma_rule_ids.strip():
        gamma_rule_ids = _parse_csv_list(args.gamma_rule_ids)
    else:
        gamma_rule_ids = []
        if len(ref_labels) >= 1:
            gamma_rule_ids.append("ref1")
        if len(ref_labels) >= 3:
            gamma_rule_ids.append("ref3")
        elif len(ref_labels) >= 2:
            gamma_rule_ids.append("ref2")
        gamma_rule_ids.append("majority")
    gamma_rule_ids = list(dict.fromkeys(gamma_rule_ids))
    unknown_gamma_rules = [r for r in gamma_rule_ids if r not in rule_by_id]
    if unknown_gamma_rules:
        raise ValueError(f"Unknown gamma rule ids: {unknown_gamma_rules}")

    exp2_rows: List[Dict[str, object]] = []
    for gamma in gamma_values:
        gamma_rows, _ = eval_rules(gamma=gamma, active_rule_ids=gamma_rule_ids)
        exp2_rows.extend(gamma_rows)

    exp2_target_rows = [
        {
            "gamma": row["gamma"],
            "rule_id": row["rule_id"],
            "rule_label": row["rule_label"],
            "target_method": target_method,
            "safe_set_size": row["safe_set_size"],
            "retained_count": row["retained_count"],
            "mixed_worst_acc": row["mixed_worst_acc"],
            "mixed_worst_mf1": row["mixed_worst_mf1"],
            "mixed_avg_acc": row["mixed_avg_acc"],
            "mixed_avg_mf1": row["mixed_avg_mf1"],
        }
        for row in exp2_rows
        if str(row.get("method", "")) == target_method
    ]

    exp2_summary_rows: List[Dict[str, object]] = []
    exp2_groups = _group_rows_by(exp2_rows, fields=("gamma", "rule_id"))
    for (gamma, rule_id), members in sorted(exp2_groups.items(), key=lambda x: (float(x[0][0]), str(x[0][1]))):
        spec = rule_by_id[str(rule_id)]
        target = next((r for r in members if str(r.get("method", "")) == target_method), None)
        row = {
            "gamma": gamma,
            "rule_id": rule_id,
            "rule_label": spec["label"],
            "ranking_worst_acc": _ranking_string(members, metric_field="mixed_worst_acc"),
            "ranking_worst_mf1": _ranking_string(members, metric_field="mixed_worst_mf1"),
        }
        if target is not None:
            row.update(
                {
                    "target_method": target_method,
                    "target_safe_set_size": target["safe_set_size"],
                    "target_retained_count": target["retained_count"],
                    "target_mixed_worst_acc": target["mixed_worst_acc"],
                    "target_mixed_worst_mf1": target["mixed_worst_mf1"],
                    "target_mixed_avg_acc": target["mixed_avg_acc"],
                    "target_mixed_avg_mf1": target["mixed_avg_mf1"],
                }
            )
        exp2_summary_rows.append(row)

    output_root = Path(args.output_root)
    csv_dir = output_root / "csv"
    tables_dir = output_root / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    p_exp1_metrics = csv_dir / "reference_rule_metrics.csv"
    p_exp1_target = csv_dir / "reference_rule_target_method.csv"
    p_exp1_summary = csv_dir / "reference_rule_summary.csv"
    p_exp1_jaccard = csv_dir / "reference_rule_jaccard.csv"
    p_exp2_metrics = csv_dir / "gamma_rule_metrics.csv"
    p_exp2_target = csv_dir / "gamma_rule_target_method.csv"
    p_exp2_summary = csv_dir / "gamma_rule_summary.csv"

    _write_csv(p_exp1_metrics, exp1_rows)
    _write_csv(p_exp1_target, exp1_target_rows)
    _write_csv(p_exp1_summary, exp1_summary_rows)
    _write_csv(p_exp1_jaccard, exp1_jaccard_rows)
    _write_csv(p_exp2_metrics, exp2_rows)
    _write_csv(p_exp2_target, exp2_target_rows)
    _write_csv(p_exp2_summary, exp2_summary_rows)

    meta_rows = [
        {"key": "target_method", "value": target_method},
        {"key": "gamma_fixed", "value": gamma_fixed},
        {"key": "gamma_values", "value": ",".join(f"{g:.2f}" for g in gamma_values)},
        {"key": "gamma_rule_ids", "value": ",".join(gamma_rule_ids)},
        {"key": "teacher_summary_csvs", "value": "|".join(str(p) for p in teacher_paths)},
        {"key": "summary_csvs", "value": "|".join(str(p) for p in summary_paths)},
        {"key": "ref_labels", "value": "|".join(ref_labels)},
        {"key": "methods", "value": "|".join(methods)},
        {
            "key": "note",
            "value": (
                "safe set varies by reference/rule; ranking stability should be inspected via *_summary.csv "
                "and *_metrics.csv rank columns."
            ),
        },
    ]
    p_meta = tables_dir / "analysis_meta.csv"
    _write_csv(p_meta, meta_rows)

    print(f"saved_exp1_metrics_csv={p_exp1_metrics}")
    print(f"saved_exp1_target_csv={p_exp1_target}")
    print(f"saved_exp1_summary_csv={p_exp1_summary}")
    print(f"saved_exp1_jaccard_csv={p_exp1_jaccard}")
    print(f"saved_exp2_metrics_csv={p_exp2_metrics}")
    print(f"saved_exp2_target_csv={p_exp2_target}")
    print(f"saved_exp2_summary_csv={p_exp2_summary}")
    print(f"saved_meta_csv={p_meta}")


if __name__ == "__main__":
    main()
