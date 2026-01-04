#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LABELS = [1, 2, 3, 4, 5]
FEATURES_7 = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
FEATURES_6 = ["x1", "x2", "x3", "x4", "x5", "x6"]


def sha1_of_row(row: np.ndarray) -> str:
    # stable hash for overlap checking
    b = row.tobytes()
    return hashlib.sha1(b).hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, cols: List[str], name: str) -> List[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")
    return cols


def coerce_numeric(df: pd.DataFrame, cols: List[str], name: str) -> None:
    # Convert and verify numeric; raise if impossible.
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[cols].isna().any(axis=1)
    if bad.any():
        n = int(bad.sum())
        raise ValueError(f"{name}: {n} rows have non-numeric or missing values in {cols}")


def check_no_inf(df: pd.DataFrame, cols: List[str], name: str) -> None:
    arr = df[cols].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: found inf/-inf in columns {cols}")


def check_label_set(df: pd.DataFrame, label_col: str, name: str) -> None:
    if label_col not in df.columns:
        raise ValueError(f"{name}: missing label column '{label_col}'")
    y = pd.to_numeric(df[label_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"{name}: label column '{label_col}' has NaNs/non-numeric values")
    uniq = sorted(y.unique().tolist())
    bad = [v for v in uniq if int(v) not in LABELS]
    if bad:
        raise ValueError(f"{name}: label column contains values outside {LABELS}: {bad}")


def check_ranges(df: pd.DataFrame, name: str) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}

    # x1..x6 in [0,10]
    for c in FEATURES_6:
        mn, mx = float(df[c].min()), float(df[c].max())
        stats[c] = {"min": mn, "max": mx}
        if mn < 0.0 - 1e-9 or mx > 10.0 + 1e-9:
            raise ValueError(f"{name}: {c} out of expected range [0,10]. min={mn}, max={mx}")

    # x7 in [0,40]
    mn, mx = float(df["x7"].min()), float(df["x7"].max())
    stats["x7"] = {"min": mn, "max": mx}
    if mn < 0.0 - 1e-9 or mx > 40.0 + 1e-9:
        raise ValueError(f"{name}: x7 out of expected range [0,40]. min={mn}, max={mx}")

    # S in [0,10]
    S = df[FEATURES_6].mean(axis=1)
    mn, mx = float(S.min()), float(S.max())
    stats["S"] = {"min": mn, "max": mx}
    if mn < 0.0 - 1e-9 or mx > 10.0 + 1e-9:
        raise ValueError(f"{name}: S out of expected range [0,10]. min={mn}, max={mx}")

    return stats


def class_counts(df: pd.DataFrame, label_col: str) -> Dict[int, int]:
    vc = df[label_col].astype(int).value_counts().to_dict()
    return {k: int(vc.get(k, 0)) for k in LABELS}


def check_expected_counts(
    df: pd.DataFrame,
    label_col: str,
    name: str,
    expected_total: int | None,
    expected_per_class: int | None,
) -> Dict[str, object]:
    counts = class_counts(df, label_col)
    n = int(len(df))

    if expected_total is not None and n != expected_total:
        raise ValueError(f"{name}: expected total {expected_total}, got {n}")

    if expected_per_class is not None:
        bad = {k: v for k, v in counts.items() if v != expected_per_class}
        if bad:
            raise ValueError(
                f"{name}: per-class counts mismatch (expected {expected_per_class} each). Got: {counts}"
            )

    return {"n": n, "per_class": counts}


def overlap_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols_for_overlap: List[str],
) -> Dict[str, object]:
    # Hash rows on a consistent float array for selected columns
    A = train_df[cols_for_overlap].to_numpy(dtype=float)
    B = test_df[cols_for_overlap].to_numpy(dtype=float)

    train_hashes = set(sha1_of_row(A[i]) for i in range(A.shape[0]))
    overlaps = []
    for i in range(B.shape[0]):
        h = sha1_of_row(B[i])
        if h in train_hashes:
            overlaps.append(i)

    return {
        "overlap_count": int(len(overlaps)),
        "overlap_test_indices_sample": overlaps[:20],
        "checked_columns": cols_for_overlap,
    }


def run_checks(
    train_path: Path,
    test_path: Path,
    label_col: str,
    expected_train_total: int | None,
    expected_test_total: int | None,
    expected_train_per_class: int | None,
    expected_test_per_class: int | None,
) -> Tuple[Dict[str, object], List[str]]:
    errors: List[str] = []
    report: Dict[str, object] = {
        "files": {"train": str(train_path), "test": str(test_path)},
        "label_col": label_col,
        "checks": {},
    }

    try:
        train_df = load_csv(train_path)
        test_df = load_csv(test_path)
        report["checks"]["load_csv"] = "PASS"
    except Exception as e:
        errors.append(str(e))
        report["checks"]["load_csv"] = f"FAIL: {e}"
        return report, errors

    try:
        require_columns(train_df, FEATURES_7 + [label_col], "TRAIN")
        require_columns(test_df, FEATURES_7 + [label_col], "TEST")
        report["checks"]["columns"] = "PASS"
    except Exception as e:
        errors.append(str(e))
        report["checks"]["columns"] = f"FAIL: {e}"
        return report, errors

    try:
        coerce_numeric(train_df, FEATURES_7 + [label_col], "TRAIN")
        coerce_numeric(test_df, FEATURES_7 + [label_col], "TEST")
        check_no_inf(train_df, FEATURES_7, "TRAIN")
        check_no_inf(test_df, FEATURES_7, "TEST")
        report["checks"]["numeric_no_nan_inf"] = "PASS"
    except Exception as e:
        errors.append(str(e))
        report["checks"]["numeric_no_nan_inf"] = f"FAIL: {e}"
        return report, errors

    try:
        check_label_set(train_df, label_col, "TRAIN")
        check_label_set(test_df, label_col, "TEST")
        report["checks"]["labels_valid"] = "PASS"
    except Exception as e:
        errors.append(str(e))
        report["checks"]["labels_valid"] = f"FAIL: {e}"
        return report, errors

    try:
        train_ranges = check_ranges(train_df, "TRAIN")
        test_ranges = check_ranges(test_df, "TEST")
        report["train_feature_ranges"] = train_ranges
        report["test_feature_ranges"] = test_ranges
        report["checks"]["ranges"] = "PASS"
    except Exception as e:
        errors.append(str(e))
        report["checks"]["ranges"] = f"FAIL: {e}"
        return report, errors

    try:
        train_counts = check_expected_counts(
            train_df, label_col, "TRAIN", expected_train_total, expected_train_per_class
        )
        test_counts = check_expected_counts(
            test_df, label_col, "TEST", expected_test_total, expected_test_per_class
        )
        report["train_counts"] = train_counts
        report["test_counts"] = test_counts
        report["checks"]["class_balance"] = "PASS"
    except Exception as e:
        errors.append(str(e))
        report["checks"]["class_balance"] = f"FAIL: {e}"
        return report, errors

    try:
        # For overlap, include features + label (strictest) or just features (more tolerant).
        cols_for_overlap = FEATURES_7 + [label_col]
        ov = overlap_check(train_df, test_df, cols_for_overlap)
        report["overlap_check"] = ov
        if ov["overlap_count"] != 0:
            raise ValueError(f"Train/Test overlap detected: {ov['overlap_count']} rows")
        report["checks"]["train_test_disjoint"] = "PASS"
    except Exception as e:
        errors.append(str(e))
        report["checks"]["train_test_disjoint"] = f"FAIL: {e}"
        return report, errors

    return report, errors


def main():
    ap = argparse.ArgumentParser(description="Sanity checks for ANFIS splits (train/test).")
    ap.add_argument("--train", required=True, help="Path to training CSV")
    ap.add_argument("--test", required=True, help="Path to test CSV")
    ap.add_argument("--label_col", default="remarks", help="Label column name (default: remarks)")

    # Defaults match your generator: 50,000 total per iter with 1/4 test => 37,500 train, 12,500 test
    # And balanced => 7,500/class train, 2,500/class test
    ap.add_argument("--expected_train_total", type=int, default=37500)
    ap.add_argument("--expected_test_total", type=int, default=12500)
    ap.add_argument("--expected_train_per_class", type=int, default=7500)
    ap.add_argument("--expected_test_per_class", type=int, default=2500)

    ap.add_argument("--out_dir", default="results/anfis_results/sanity", help="Where to save sanity report JSON")
    args = ap.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report, errors = run_checks(
        train_path=train_path,
        test_path=test_path,
        label_col=args.label_col,
        expected_train_total=args.expected_train_total,
        expected_test_total=args.expected_test_total,
        expected_train_per_class=args.expected_train_per_class,
        expected_test_per_class=args.expected_test_per_class,
    )

    report["status"] = "PASS" if not errors else "FAIL"
    report["errors"] = errors

    # Save report
    tag = f"{train_path.stem}__{test_path.stem}"
    report_path = out_dir / f"sanity_{tag}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary
    print("\n=== ANFIS Sanity Checks ===")
    print(f"Train: {train_path}")
    print(f"Test : {test_path}")
    print(f"Status: {report['status']}")
    print(f"Report saved to: {report_path.resolve()}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("\nKey counts:")
    print("Train per-class:", report["train_counts"]["per_class"])
    print("Test  per-class:", report["test_counts"]["per_class"])
    print("Overlap count:", report["overlap_check"]["overlap_count"])
    print("\nAll checks PASSED.")


if __name__ == "__main__":
    main()
