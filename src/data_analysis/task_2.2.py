# Task 2.2 — Target (remarks) Analysis 

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "academicPerformanceData.xlsx"

OUT_TABLES = PROJECT_ROOT / "results" / "data_analysis" / "tables"
OUT_PLOTS  = PROJECT_ROOT / "results" / "data_analysis" / "plots"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

# IMPORTANT: real header is on row 2 
df = pd.read_excel(DATA_FILE, header=1)

df.columns = [str(c).strip() for c in df.columns]

# Find target column robustly 
norm_cols = {c.lower(): c for c in df.columns}
if "remarks" not in norm_cols:
    raise ValueError(f"'remarks' not found. Columns: {list(df.columns)}")

TARGET_COL = norm_cols["remarks"]

# Distribution
dist = (
    df[TARGET_COL]
    .astype(str)
    .str.strip()
    .value_counts(dropna=False)
    .rename_axis("remarks")
    .reset_index(name="count")
)

# Save
out_path = OUT_TABLES / "remarks_distribution.csv"
dist.to_csv(out_path, index=False)

print("\n=== Task 2.2 Complete ===")
print(f"Columns detected: {list(df.columns)}")
print(f"Target column: {TARGET_COL}")
print(f"Number of classes: {len(dist)}\n")
print(dist.to_string(index=False))
print(f"\nSaved to: {out_path}")
