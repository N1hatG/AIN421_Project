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
# x7 only definition
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


def x7_boundary_memberships(x7):
    """
    'Near boundary' fuzzy sets around the A2 boundaries:
      b12=10.5, b23=19.5, b34=28.0, b45=34.0
    Narrow triangles centered on the boundary.
    """
    b12, b23, b34, b45 = 10.5, 19.5, 28.0, 34.0
    w12, w23, w34, w45 = 2.0, 2.5, 2.5, 2.0

    nb12 = trimf(x7, b12 - w12, b12, b12 + w12)
    nb23 = trimf(x7, b23 - w23, b23, b23 + w23)
    nb34 = trimf(x7, b34 - w34, b34, b34 + w34)
    nb45 = trimf(x7, b45 - w45, b45, b45 + w45)

    return nb12, nb23, nb34, nb45


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


def fis_predict_x7_only(x7_values, use_refinement: bool):
    """
    Mamdani inference for each x7 value.

    core5:      5 core rules (VL->1, L->2, M->3, H->4, VH->5)
    refined10:  core5 + 5 near-boundary refinement rules
    """
    x7_values = np.asarray(x7_values, dtype=float)

    # output universe
    y = np.linspace(0.5, 5.5, 2001)
    out_mus = output_memberships(y)  # (mu1..mu5), each is array over y

    preds = []
    crisp_vals = []

    for x7 in x7_values:
        # Main memberships (5 core sets)
        mu_vl, mu_l, mu_m, mu_h, mu_vh = x7_memberships(np.array([x7]))
        mu_vl, mu_l, mu_m, mu_h, mu_vh = float(mu_vl[0]), float(mu_l[0]), float(mu_m[0]), float(mu_h[0]), float(mu_vh[0])

        rule_firing = [mu_vl, mu_l, mu_m, mu_h, mu_vh]
        rule_outputs = [out_mus[0], out_mus[1], out_mus[2], out_mus[3], out_mus[4]]

        all_firing = rule_firing
        all_outputs = rule_outputs

        if use_refinement:
            nb12, nb23, nb34, nb45 = x7_boundary_memberships(np.array([x7]))
            nb12, nb23, nb34, nb45 = float(nb12[0]), float(nb23[0]), float(nb34[0]), float(nb45[0])

            ref_firing = [
                nb12,                           # class1
                nb23,                           # class2
                nb34,                           # class4
                nb45,                           # class5
                float(np.minimum(nb23, mu_m)),  # class3 (near 19.5 AND medium-ish)
            ]
            ref_outputs = [
                out_mus[0],  # class1
                out_mus[1],  # class2
                out_mus[3],  # class4
                out_mus[4],  # class5
                out_mus[2],  # class3
            ]

            all_firing = rule_firing + ref_firing
            all_outputs = rule_outputs + ref_outputs

        # Implication (min) + aggregation (max)
        clipped = [np.minimum(all_firing[i], all_outputs[i]) for i in range(len(all_firing))]
        aggregated = np.maximum.reduce(clipped)

        # Defuzzify (centroid)
        y_star = centroid_defuzz(y, aggregated)
        crisp_vals.append(y_star)

        # Convert to class label
        if np.isnan(y_star):
            pred_class = 3
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
        default="refined10",
        choices=["core5", "refined10"],
        help="core5 = 5 core rules, refined10 = 5 core + 5 refinement rules"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    input_path = root / args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    cols = {c.strip().lower(): c for c in df.columns}
    if "remarks" not in cols or "x7" not in cols:
        raise ValueError(f"Need 'x7' and 'remarks' columns. Found: {list(df.columns)}")

    x7 = df[cols["x7"]].values
    y_true = df[cols["remarks"]].astype(int).values

    use_refinement = (args.mode == "refined10")
    y_pred, y_crisp = fis_predict_x7_only(x7, use_refinement=use_refinement)

    acc = float((y_pred == y_true).mean())

    out_dir = root / "results" / "fis_results" / "fis-1"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.mode

    # Confusion matrix save
    cm = confusion_matrix(y_true, y_pred)
    cm_path = out_dir / f"confusion_{input_path.stem}_fis_x7_{tag}.csv"
    cm.astype(int).to_csv(cm_path)

    # Predictions save (results-only formatting)
    out_df = df.copy()
    if "S" in out_df.columns:
        out_df["S"] = out_df["S"].astype(float).round(3)

    out_df["fis_x7_pred"] = y_pred
    out_df["fis_x7_crisp"] = np.round(y_crisp, 3)

    out_file = out_dir / f"predictions_{input_path.stem}_fis_x7_{tag}.csv"
    out_df.to_csv(out_file, index=False)

    # Console output
    print(f"\n=== FIS-1 (x7 only) mode={tag} ===")
    print("Input:", input_path)
    print(f"Accuracy: {acc:.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):\n")
    print(cm.astype(int))
    print("\nSaved confusion matrix to:", cm_path)
    print("\nSaved predictions to:", out_file)


if __name__ == "__main__":
    main()
