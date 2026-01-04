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

    # exact per-class sizes
    n_per_class = 10_000
    n_test_per_class = int(round(n_per_class * test_ratio))   # 2500 if ratio=0.25
    n_train_per_class = n_per_class - n_test_per_class        # 7500

    train_parts = []
    test_parts = []

    for cls in classes:
        cls_df = df[df[tcol] == cls]
        if len(cls_df) < n_per_class:
            raise ValueError(f"Class {cls} has only {len(cls_df)} rows (<{n_per_class}).")

        # sample exactly 10,000 from this class
        pick_idx = rng.choice(cls_df.index.to_numpy(), size=n_per_class, replace=False)
        picked = cls_df.loc[pick_idx].copy()

        # shuffle within class and split exactly 7500/2500
        perm = rng.permutation(n_per_class)
        test_idx = perm[:n_test_per_class]
        train_idx = perm[n_test_per_class:]

        test_parts.append(picked.iloc[test_idx])
        train_parts.append(picked.iloc[train_idx])

    # combine & shuffle (optional, but nice)
    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df  = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # add engineered feature S
    train_df = add_S(train_df)
    test_df  = add_S(test_df)

    train_path = out_dir / f"anfis_iter{iter_id}_train.csv"
    test_path  = out_dir / f"anfis_iter{iter_id}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "type": "ANFIS",
        "iter_id": iter_id,
        "seed": seed,
        "total_rows": int(n_per_class * len(classes)),  # 50,000
        "rows_per_class": n_per_class,                  # 10,000
        "test_ratio": test_ratio,
        "train_rows": int(train_df.shape[0]),           # 37,500
        "test_rows": int(test_df.shape[0]),             # 12,500
        "train_rows_per_class": n_train_per_class,      # 7,500
        "test_rows_per_class": n_test_per_class,        # 2,500
        "classes": [int(c) if str(c).isdigit() else str(c) for c in classes],
        "features": FEATURES,
        "engineered_features": ["S (mean of x1..x6)"],
        "target": tcol,
        "source": "academicPerformanceData.xlsx (header=1)",
        "note": "Class-wise split: exactly 10,000 per class, then 7,500 train / 2,500 test per class."
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
