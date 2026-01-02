# Task 2.3 — Feature Range & Distribution Analysis

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------ Paths ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "academicPerformanceData.xlsx"

OUT_TABLES = PROJECT_ROOT / "results" / "data_analysis" / "tables"
OUT_PLOTS_BASE = PROJECT_ROOT / "results" / "data_analysis" / "plots"
OUT_DIST_PLOTS = OUT_PLOTS_BASE / "feature_distributions"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_DIST_PLOTS.mkdir(parents=True, exist_ok=True)

# ------------------ Load ------------------
# IMPORTANT: your file's real header is on row 1 (0-indexed)
df = pd.read_excel(DATA_FILE, header=1)
df.columns = [str(c).strip() for c in df.columns]

# ------------------ Select numeric features ------------------
# Exclude target if it exists
target_col = None
for c in df.columns:
    if str(c).strip().lower() == "remarks":
        target_col = c
        break

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# If everything is numeric but remarks is encoded as number (unlikely), remove it safely:
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

if not numeric_cols:
    raise ValueError("No numeric columns detected. Check your Excel parsing / header row.")

# ------------------ Summary statistics ------------------
# Use describe + extra missing info
desc = df[numeric_cols].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
desc.rename(columns={"50%": "median"}, inplace=True)

missing = df[numeric_cols].isna().sum().rename("missing_count")
missing_pct = (df[numeric_cols].isna().mean() * 100).rename("missing_pct")

stats = desc.join(missing).join(missing_pct)
stats.reset_index(inplace=True)
stats.rename(columns={"index": "feature"}, inplace=True)

stats_path = OUT_TABLES / "feature_statistics.csv"
stats.to_csv(stats_path, index=False)

# ------------------ Plots per feature ------------------
# Histogram and Boxplot per numeric feature
for col in numeric_cols:
    series = df[col].dropna()

    # Histogram
    plt.figure()
    plt.hist(series, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT_DIST_PLOTS / f"{col}_hist.png", dpi=200)
    plt.close()

    # Boxplot
    plt.figure()
    plt.boxplot(series, vert=True)
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(OUT_DIST_PLOTS / f"{col}_box.png", dpi=200)
    plt.close()

print("\n=== Task 2.3 Complete ===")
print(f"Numeric features plotted: {len(numeric_cols)}")
print(f"Saved stats table: {stats_path}")
print(f"Saved plots folder: {OUT_DIST_PLOTS}")
