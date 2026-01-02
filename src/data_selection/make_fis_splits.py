import numpy as np
import pandas as pd
from pathlib import Path
from utils import project_root_from, load_dataset, ensure_dirs, save_meta

FEATURES = ["x1","x2","x3","x4","x5","x6","x7"]
TARGET = "remarks"

def add_S(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["S"] = df[["x1","x2","x3","x4","x5","x6"]].mean(axis=1)
    return df

def normalize_target_col(df: pd.DataFrame) -> str:
    norm_cols = {c.lower(): c for c in df.columns}
    if TARGET not in norm_cols:
        raise ValueError(f"'{TARGET}' not found. Columns: {list(df.columns)}")
    return norm_cols[TARGET]

def sample_20_per_class_disjoint(
    df: pd.DataFrame,
    tcol: str,
    seed: int,
    forbidden_index: set
) -> tuple[pd.DataFrame, set]:
    """
    Samples 20 rows per class without replacement, and ensures sampled indices
    do NOT intersect forbidden_index. Returns (sampled_df, sampled_index_set).
    """
    rng = np.random.default_rng(seed)
    classes = sorted(df[tcol].unique())

    picked_all = []
    picked_idx_set = set()

    for cls in classes:
        cls_df = df[df[tcol] == cls]

        # exclude indices for disjointness
        available = cls_df.loc[~cls_df.index.isin(forbidden_index)]

        if len(available) < 20:
            raise ValueError(
                f"Cannot sample 20 disjoint rows for class {cls}. "
                f"Available after excluding run1: {len(available)}"
            )

        pick_idx = rng.choice(available.index.to_numpy(), size=20, replace=False)
        picked_idx_set.update(pick_idx.tolist())
        picked_all.append(df.loc[pick_idx])

    out = pd.concat(picked_all, axis=0)
    out = out.sample(frac=1.0, random_state=seed)  # shuffle rows
    return out, picked_idx_set

def make_fis_runs_disjoint(df: pd.DataFrame, out_dir: Path):
    tcol = normalize_target_col(df)

    # Run 1
    run1_seed = 42101
    run1_df, run1_idx = sample_20_per_class_disjoint(
        df, tcol=tcol, seed=run1_seed, forbidden_index=set()
    )
    run1_df = add_S(run1_df)
    run1_path = out_dir / "fis_run1.csv"
    run1_df.to_csv(run1_path, index=False)

    save_meta(out_dir / "meta_fis_run1.json", {
        "type": "FIS",
        "run_id": 1,
        "seed": run1_seed,
        "total_rows": int(run1_df.shape[0]),
        "rows_per_class": 20,
        "classes": sorted(df[tcol].unique()),
        "features": FEATURES,
        "engineered_features": ["S (mean of x1..x6)"],
        "target": tcol,
        "disjoint_from_other_runs": True
    })

    # Run 2 (disjointed from 1)
    run2_seed = 42102
    run2_df, run2_idx = sample_20_per_class_disjoint(
        df, tcol=tcol, seed=run2_seed, forbidden_index=run1_idx
    )
    run2_df = add_S(run2_df)
    run2_path = out_dir / "fis_run2.csv"
    run2_df.to_csv(run2_path, index=False)

    save_meta(out_dir / "meta_fis_run2.json", {
        "type": "FIS",
        "run_id": 2,
        "seed": run2_seed,
        "total_rows": int(run2_df.shape[0]),
        "rows_per_class": 20,
        "classes": sorted(df[tcol].unique()),
        "features": FEATURES,
        "engineered_features": ["S (mean of x1..x6)"],
        "target": tcol,
        "disjoint_from_other_runs": True,
        "note": "fis_run2 is sampled from rows excluding fis_run1 indices"
    })

    # Extra safety check (should be 0)
    overlap = set(run1_idx).intersection(run2_idx)
    print("FIS disjoint sampling complete.")
    print("Overlap (should be 0):", len(overlap))
    print("Saved:", run1_path)
    print("Saved:", run2_path)

def main():
    ROOT = project_root_from(__file__)
    data_file = ROOT / "data" / "raw" / "academicPerformanceData.xlsx"
    splits_dir = ROOT / "data" / "splits" / "fis"
    ensure_dirs(splits_dir)

    df = load_dataset(data_file)
    make_fis_runs_disjoint(df, splits_dir)

if __name__ == "__main__":
    main()
