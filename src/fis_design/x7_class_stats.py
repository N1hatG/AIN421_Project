import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "academicPerformanceData.xlsx"

OUT_DIR = ROOT / "results" / "fis_design"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA, header=1)
df.columns = df.columns.str.strip()

TARGET = "remarks"

# Per-class statistics
stats = (
    df.groupby(TARGET)["x7"]
    .describe(percentiles=[0.25, 0.5, 0.75])
)

# Round stats for saving
stats_rounded = stats.round(3)
stats_path = OUT_DIR / "x7_per_class_statistics.csv"
stats_rounded.to_csv(stats_path)

# Compute class-separating boundaries from MEDIANS
med = stats["50%"]
bounds = [(med.iloc[i] + med.iloc[i+1]) / 2 for i in range(len(med)-1)]

bounds_df = pd.DataFrame({
    "boundary_between_classes": [f"{i+1} | {i+2}" for i in range(len(bounds))],
    "x7_value": [round(b, 3) for b in bounds]
})

bounds_path = OUT_DIR / "x7_proposed_boundaries.csv"
bounds_df.to_csv(bounds_path, index=False)

# Console output
print("\n=== x7 per-class statistics ===\n")
print(stats.round(3))

print("\n=== Proposed x7 boundaries (between classes) ===\n")
for i, b in enumerate(bounds, 1):
    print(f"B{i}|{i+1} = {b:.3f}")

print("\nSaved files:")
print(stats_path)
print(bounds_path)
