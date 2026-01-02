# Task 2.5 — Feature vs Target (Readable Distributions)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "academicPerformanceData.xlsx"

OUT_PLOTS = PROJECT_ROOT / "results" / "data_analysis" / "plots" / "feature_vs_remarks"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

# Load (header is on row 1)
df = pd.read_excel(DATA_FILE, header=1)
df.columns = [c.strip() for c in df.columns]

# Target
TARGET = "remarks"
df[TARGET] = df[TARGET].astype(int)

# Fixed feature order
FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]

classes = sorted(df[TARGET].unique())

for feature in FEATURES:

    fig, axes = plt.subplots(1, len(classes), figsize=(18, 3), sharey=True)
    fig.suptitle(f"{feature} distribution by remarks class", fontsize=14)

    min_val = df[feature].min()
    max_val = df[feature].max()

    for ax, cls in zip(axes, classes):
        subset = df[df[TARGET] == cls][feature]

        ax.hist(subset, bins=30)
        ax.set_title(f"Class {cls}")
        ax.set_xlim(min_val, max_val)
        ax.set_xlabel(feature)

    axes[0].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"{feature}_by_remarks_hist.png", dpi=200)
    plt.close()

print("\n=== Task 2.5 Complete ===")
print("Saved readable feature-vs-target plots to:")
print(OUT_PLOTS)
