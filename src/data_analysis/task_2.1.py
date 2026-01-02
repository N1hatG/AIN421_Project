# Task 2.1 — Load & Inspect Dataset 

import pandas as pd
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path.cwd()  
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "academicPerformanceData.xlsx"

OUT_TABLES = PROJECT_ROOT / "results" / "data_analysis" / "tables"
OUT_PLOTS  = PROJECT_ROOT / "results" / "data_analysis" / "plots" 
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

# --- Load ---
df = pd.read_excel(DATA_FILE)

# Optional: clean column names (keeps content same, just removes accidental spaces)
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

# --- Basic overview ---
overview = pd.DataFrame({
    "n_rows": [df.shape[0]],
    "n_cols": [df.shape[1]]
})

dtypes = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in df.dtypes]
})

missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_pct": (df.isna().mean().values * 100).round(3)
}).sort_values("missing_count", ascending=False)

# --- Save tables ---
overview.to_csv(OUT_TABLES / "dataset_overview.csv", index=False)
dtypes.to_csv(OUT_TABLES / "column_dtypes.csv", index=False)
missing.to_csv(OUT_TABLES / "missing_values.csv", index=False)

# --- Print quick summary for you ---
print("Loaded:", DATA_FILE)
print("Shape:", df.shape)
print("\nTop missing-value columns:")
print(missing.head(10).to_string(index=False))
print("\nSaved to:")
print(" -", OUT_TABLES / "dataset_overview.csv")
print(" -", OUT_TABLES / "column_dtypes.csv")
print(" -", OUT_TABLES / "missing_values.csv")
