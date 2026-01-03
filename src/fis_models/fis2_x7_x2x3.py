import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------
# Membership functions
# --------------------
def trimf(x, a, b, c):
    """Triangular membership function."""
    x = np.asarray(x)
    mu = np.zeros_like(x, dtype=float)
    left = (a < x) & (x < b)
    mu[left] = (x[left] - a) / (b - a)
    mu[x == b] = 1.0
    right = (b < x) & (x < c)
    mu[right] = (c - x[right]) / (c - b)
    return np.clip(mu, 0.0, 1.0)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    x = np.asarray(x)
    mu = np.zeros_like(x, dtype=float)

    if b != a:
        left = (a < x) & (x < b)
        mu[left] = (x[left] - a) / (b - a)

    mid = (b <= x) & (x <= c)
    mu[mid] = 1.0

    if d != c:
        right = (c < x) & (x < d)
        mu[right] = (d - x[right]) / (d - c)

    return np.clip(mu, 0.0, 1.0)


def centroid_defuzz(y_universe, aggregated_mu):
    """Centroid defuzzification."""
    num = np.sum(y_universe * aggregated_mu)
    den = np.sum(aggregated_mu)
    if den == 0:
        return np.nan
    return num / den


# ------------------
# Input memberships
# ------------------
def x7_memberships(x7):
    """
    A2-based sets using your stats:
    boundaries: 10.5, 19.5, 28.0, 34.0
    centers: 6, 15, 24, 32, 36
    """
    mu_vl = trapmf(x7, 0.0, 0.0, 6.0, 10.5)        # Very Low
    mu_l  = trimf(x7, 10.5, 15.0, 19.5)            # Low
    mu_m  = trimf(x7, 19.5, 24.0, 28.0)            # Medium
    mu_h  = trimf(x7, 28.0, 32.0, 34.0)            # High
    mu_vh = trapmf(x7, 34.0, 36.0, 40.0, 40.0)     # Very High
    return mu_vl, mu_l, mu_m, mu_h, mu_vh


def x2x3_memberships(v):
    """
    x2, x3 are in [0,10] and behave similar to x1..x6.
    3 fuzzy sets:
      Low, Medium, High
    """
    v = np.asarray(v, dtype=float)

    mu_low = trapmf(v, 0.0, 0.0, 2.0, 4.0)
    mu_med = trimf(v, 3.0, 5.0, 7.0)
    mu_high = trapmf(v, 6.0, 8.0, 10.0, 10.0)

    return mu_low, mu_med, mu_high


def output_memberships(y):
    """
    Output remarks universe ~ [0.5, 5.5]
    Define 5 fuzzy sets around 1..5.
    """
    mu1 = trapmf(y, 0.5, 0.5, 1.0, 1.5)
    mu2 = trimf(y, 1.5, 2.0, 2.5)
    mu3 = trimf(y, 2.5, 3.0, 3.5)
    mu4 = trimf(y, 3.5, 4.0, 4.5)
    mu5 = trapmf(y, 4.5, 5.0, 5.5, 5.5)
    return mu1, mu2, mu3, mu4, mu5


# -----------------------
# Helpers: gate + weighting
# -----------------------
def relax_gate(overlap_val, side_strength, relax_alpha):
    """
    Softly relax the overlap gate so refinements can activate more often.

    gate = max(overlap, relax_alpha * side_strength)

    Example: for 1|2 boundary, side_strength can be mu_l (the "right" side).
    """
    return max(float(overlap_val), float(relax_alpha) * float(side_strength))


def weight_fire(fire_strength, ref_weight):
    """
    Boost refinement influence by multiplying firing strength and clipping to 1.
    """
    return float(min(1.0, float(ref_weight) * float(fire_strength)))


# -----------------------
# FIS prediction
# -----------------------
def fis_predict_x7_x2_x3(
    x7_values,
    x2_values,
    x3_values,
    mode="overlap10",
    ref_weight=2.0,
    gate_relax=0.25,
):
    """
    Modes:
      - core5: only 5 core rules based on x7
      - overlap10: 5 core rules + 5 overlap-gated directional refinement rules

    New controls:
      - ref_weight: multiplies refinement firing strengths (clipped to 1.0)
      - gate_relax: relaxes overlap gating: gate = max(over, gate_relax * side_strength)
    """

    x7_values = np.asarray(x7_values, dtype=float)
    x2_values = np.asarray(x2_values, dtype=float)
    x3_values = np.asarray(x3_values, dtype=float)

    y = np.linspace(0.5, 5.5, 2001)
    out_mus = output_memberships(y)

    preds = []
    crisp_vals = []

    for x7, x2, x3 in zip(x7_values, x2_values, x3_values):
        mu_vl, mu_l, mu_m, mu_h, mu_vh = x7_memberships(np.array([x7]))
        mu_vl, mu_l, mu_m, mu_h, mu_vh = float(mu_vl[0]), float(mu_l[0]), float(mu_m[0]), float(mu_h[0]), float(mu_vh[0])

        x2_low, x2_med, x2_high = x2x3_memberships(np.array([x2]))
        x3_low, x3_med, x3_high = x2x3_memberships(np.array([x3]))
        x2_low, x2_med, x2_high = float(x2_low[0]), float(x2_med[0]), float(x2_high[0])
        x3_low, x3_med, x3_high = float(x3_low[0]), float(x3_med[0]), float(x3_high[0])

        low_any = max(x2_low, x3_low)      # (x2 Low OR x3 Low)
        high_any = max(x2_high, x3_high)   # (x2 High OR x3 High)

        # ---- Core 5 rules (x7 only) ----
        rule_firing = [mu_vl, mu_l, mu_m, mu_h, mu_vh]
        rule_outputs = [out_mus[0], out_mus[1], out_mus[2], out_mus[3], out_mus[4]]

        all_firing = list(rule_firing)
        all_outputs = list(rule_outputs)

        # ---- Overlap-gated refinements (5 rules) ----
        if mode.lower() == "overlap10":
            # raw overlaps
            over12 = min(mu_vl, mu_l)
            over23 = min(mu_l,  mu_m)
            over34 = min(mu_m,  mu_h)
            over45 = min(mu_h,  mu_vh)

            # relaxed gates (use the "right-side" strength to relax)
            # 1|2: use mu_l
            # 2|3: use mu_m
            # 3|4: use mu_h
            # 4|5: use mu_vh
            gate12 = relax_gate(over12, mu_l, gate_relax)
            gate23 = relax_gate(over23, mu_m, gate_relax)
            gate34 = relax_gate(over34, mu_h, gate_relax)
            gate45 = relax_gate(over45, mu_vh, gate_relax)

            # 5 refinement rules (directional, gated + weighted)
            # R6: IF (1|2 gate) AND high_any -> class2
            r6 = weight_fire(min(gate12, high_any), ref_weight)

            # R7: IF (2|3 gate) AND low_any  -> class2
            r7 = weight_fire(min(gate23, low_any), ref_weight)

            # R8: IF (2|3 gate) AND high_any -> class3
            r8 = weight_fire(min(gate23, high_any), ref_weight)

            # R9: IF (3|4 gate) AND high_any -> class4
            r9 = weight_fire(min(gate34, high_any), ref_weight)

            # R10: IF (4|5 gate) AND high_any -> class5
            r10 = weight_fire(min(gate45, high_any), ref_weight)

            ref_firing = [r6, r7, r8, r9, r10]
            ref_outputs = [
                out_mus[1],  # class2
                out_mus[1],  # class2
                out_mus[2],  # class3
                out_mus[3],  # class4
                out_mus[4],  # class5
            ]

            all_firing += ref_firing
            all_outputs += ref_outputs

        # Inference: implication=min, aggregation=max
        clipped = [np.minimum(all_firing[i], all_outputs[i]) for i in range(len(all_firing))]
        aggregated = np.maximum.reduce(clipped)

        y_star = centroid_defuzz(y, aggregated)
        crisp_vals.append(y_star)

        if np.isnan(y_star):
            pred_class = 3  # safe fallback
        else:
            pred_class = int(np.clip(np.rint(y_star), 1, 5))

        preds.append(pred_class)

    return np.array(preds, dtype=int), np.array(crisp_vals, dtype=float)


# ----------
# Evaluation
# ----------
def confusion_matrix(y_true, y_pred, labels=(1, 2, 3, 4, 5)):
    cm = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
        rownames=["true"],
        colnames=["pred"],
        dropna=False
    )

    for l in labels:
        if l not in cm.index:
            cm.loc[l] = 0
        if l not in cm.columns:
            cm[l] = 0
    cm = cm.sort_index().reindex(sorted(cm.columns), axis=1)
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/splits/fis/fis_run1.csv",
        help="Path to FIS split CSV (e.g., data/splits/fis/fis_run1.csv)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overlap10",
        choices=["core5", "overlap10"],
        help="core5 = only x7 core rules, overlap10 = core+overlap-gated refinement"
    )
    parser.add_argument(
        "--ref_weight",
        type=float,
        default=2.0,
        help="Multiplier for refinement rule firing strengths (clipped to 1.0)."
    )
    parser.add_argument(
        "--gate_relax",
        type=float,
        default=0.25,
        help="Relaxation factor for overlap gate: gate=max(overlap, gate_relax*side_strength)."
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    input_path = root / args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    cols = {c.strip().lower(): c for c in df.columns}
    needed = ["x7", "x2", "x3", "remarks"]
    if any(k not in cols for k in needed):
        raise ValueError(f"Need columns {needed}. Found: {list(df.columns)}")

    x7 = df[cols["x7"]].values
    x2 = df[cols["x2"]].values
    x3 = df[cols["x3"]].values
    y_true = df[cols["remarks"]].astype(int).values

    y_pred, y_crisp = fis_predict_x7_x2_x3(
        x7, x2, x3,
        mode=args.mode,
        ref_weight=args.ref_weight,
        gate_relax=args.gate_relax
    )
    acc = (y_pred == y_true).mean()

    # Save outputs under results/fis_results/fis-2
    out_dir = root / "results" / "fis_results" / "fis-2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "core5":
        tag = f"fis2_x7_x2x3_{args.mode}"
    else:
        tag = f"fis2_x7_x2x3_{args.mode}_w{args.ref_weight:g}_g{args.gate_relax:g}"

    cm = confusion_matrix(y_true, y_pred)
    cm_path = out_dir / f"confusion_{input_path.stem}_{tag}.csv"
    cm.astype(int).to_csv(cm_path)

    out_df = df.copy()

    # only format S for output file (do not change source data)
    if "S" in out_df.columns:
        out_df["S"] = out_df["S"].astype(float).round(3)

    out_df["fis2_pred"] = y_pred

    # crisp: round for file, fill NaN if any (can happen if aggregated=0)
    crisp_fixed = np.where(np.isnan(y_crisp), 3.0, y_crisp)
    out_df["fis2_crisp"] = np.round(crisp_fixed, 3)

    out_file = out_dir / f"predictions_{input_path.stem}_{tag}.csv"
    out_df.to_csv(out_file, index=False)

    print(f"\n=== FIS-2 (x7 + x2 + x3) mode={args.mode} ===")
    if args.mode != "core5":
        print(f"ref_weight={args.ref_weight}, gate_relax={args.gate_relax}")
    print("Input:", input_path)
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):\n")
    print(cm.astype(int))
    print("\nSaved confusion matrix to:", cm_path)
    print("Saved predictions to:", out_file)


if __name__ == "__main__":
    main()
