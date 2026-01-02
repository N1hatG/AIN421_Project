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

def make_anfis_iter(df: pd.DataFrame, iter_id: int, seed: int, out_dir: Path, test_ratio: float = 0.25):
    rng = np.random.default_rng(seed)

    norm_cols = {c.lower(): c for c in df.columns}
    if TARGET not in norm_cols:
        raise ValueError(f"'{TARGET}' not found. Columns: {list(df.columns)}")
    tcol = norm_cols[TARGET]

    classes = sorted(df[tcol].unique())

    # Sample 10,000 per class
    parts = []
    for cls in classes:
        cls_df = df[df[tcol] == cls]
        if len(cls_df) < 10_000:
            raise ValueError(f"Class {cls} has only {len(cls_df)} rows (<10,000).")
        pick_idx = rng.choice(cls_df.index.to_numpy(), size=10_000, replace=False)
        parts.append(df.loc[pick_idx])

    full = pd.concat(parts, axis=0)
    full = full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    full = add_S(full)

    # Train/test split by index
    n = full.shape[0]
    n_test = int(round(n * test_ratio))
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_df = full.iloc[train_idx].reset_index(drop=True)
    test_df  = full.iloc[test_idx].reset_index(drop=True)

    train_path = out_dir / f"anfis_iter{iter_id}_train.csv"
    test_path  = out_dir / f"anfis_iter{iter_id}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "type": "ANFIS",
        "iter_id": iter_id,
        "seed": seed,
        "total_rows": int(n),
        "rows_per_class": 10000,
        "test_ratio": test_ratio,
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "classes": [int(c) if str(c).isdigit() else str(c) for c in classes],
        "features": FEATURES,
        "engineered_features": ["S (mean of x1..x6)"],
        "target": tcol,
        "source": "academicPerformanceData.xlsx (header=1)",
        "note": "Use same train/test splits for ANFIS-7D, ANFIS-x7-only, ANFIS-(S+x7) ablations."
    }
    save_meta(out_dir / f"meta_anfis_iter{iter_id}.json", meta)

def main():
    ROOT = project_root_from(__file__)
    data_file = ROOT / "data" / "raw" / "academicPerformanceData.xlsx"
    splits_dir = ROOT / "data" / "splits" / "anfis"
    ensure_dirs(splits_dir)

    df = load_dataset(data_file)

    make_anfis_iter(df, iter_id=1, seed=42111, out_dir=splits_dir, test_ratio=0.25)
    make_anfis_iter(df, iter_id=2, seed=42112, out_dir=splits_dir, test_ratio=0.25)

    print("ANFIS splits created in:", splits_dir)

if __name__ == "__main__":
    main()
