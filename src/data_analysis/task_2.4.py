# Task 2.4 — Correlation Analysis + Heatmap

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#  Paths 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "academicPerformanceData.xlsx"

OUT_TABLES = PROJECT_ROOT / "results" / "data_analysis" / "tables"
OUT_PLOTS  = PROJECT_ROOT / "results" / "data_analysis" / "plots"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

#  Load 
df = pd.read_excel(DATA_FILE, header=1)
df.columns = [str(c).strip() for c in df.columns]

#  Select numeric columns 
numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] < 2:
    raise ValueError(
        f"Need at least 2 numeric columns for correlation. Found {numeric_df.shape[1]}."
    )

#  Correlation matrix 
corr = numeric_df.corr(numeric_only=True)

corr_path = OUT_TABLES / "correlation_matrix.csv"
corr.to_csv(corr_path, index=True)

#  Heatmap  
plt.figure(figsize=(10, 8))
plt.imshow(corr.values, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()

heatmap_path = OUT_PLOTS / "correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=250)
plt.close()

print("\n=== Task 2.4 Complete ===")
print(f"Numeric columns used: {numeric_df.shape[1]}")
print(f"Saved correlation matrix: {corr_path}")
print(f"Saved heatmap: {heatmap_path}")
