#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- basic fuzzy membership helpers ----------
def tri(x, a, b, c):
    """Triangular membership."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    # rising
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a + 1e-12)
    # falling
    idx = (b <= x) & (x < c)
    y[idx] = (c - x[idx]) / (c - b + 1e-12)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)

def trap(x, a, b, c, d):
    """Trapezoidal membership."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    # rising
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a + 1e-12)
    # top
    idx = (b <= x) & (x <= c)
    y[idx] = 1.0
    # falling
    idx = (c < x) & (x < d)
    y[idx] = (d - x[idx]) / (d - c + 1e-12)
    return np.clip(y, 0.0, 1.0)

def clip01(v):
    return np.minimum(1.0, np.maximum(0.0, v))

# ---------- FIS-3 ----------
def compute_memberships_x7(x7):
    # boundaries from your notes
    b12, b23, b34, b45 = 10.5, 19.5, 28.0, 34.0

    # simple choices consistent with your earlier idea
    VL = trap(x7, 0.0, 0.0, 6.0, b12)
    L  = tri(x7, b12, 15.0, b23)
    M  = tri(x7, b23, 24.0, b34)
    H  = tri(x7, b34, 32.0, b45)
    VH = trap(x7, b45, 36.0, 40.0, 40.0)

    return {"VL": VL, "L": L, "M": M, "H": H, "VH": VH}

def compute_memberships_S(S):
    # S in [0,10]
    Low  = trap(S, 0.0, 0.0, 2.5, 5.0)
    Med  = tri(S, 2.5, 5.0, 7.5)
    High = trap(S, 5.0, 7.5, 10.0, 10.0)
    return {"Low": Low, "Med": Med, "High": High}

def infer_fis3(df, mode="overlap10", ref_weight=1.0, gate_relax=0.25):
    x7 = df["x7"].to_numpy(dtype=float)
    S  = df["S"].to_numpy(dtype=float)

    mx7 = compute_memberships_x7(x7)
    mS  = compute_memberships_S(S)

    # overlaps (ambiguity zones)
    over12 = np.minimum(mx7["VL"], mx7["L"])
    over23 = np.minimum(mx7["L"],  mx7["M"])
    over34 = np.minimum(mx7["M"],  mx7["H"])
    over45 = np.minimum(mx7["H"],  mx7["VH"])

    # optional relaxed gating (same spirit as your FIS-2 tweak)
    # side_strength = max(left,right), so overlap gate can be boosted
    def relaxed_gate(over, left, right):
        side = np.maximum(left, right)
        return np.maximum(over, gate_relax * side)

    g12 = relaxed_gate(over12, mx7["VL"], mx7["L"])
    g23 = relaxed_gate(over23, mx7["L"],  mx7["M"])
    g34 = relaxed_gate(over34, mx7["M"],  mx7["H"])
    g45 = relaxed_gate(over45, mx7["H"],  mx7["VH"])

    # We build a simple "class score" aggregation:
    # each rule contributes firing_strength to its target class.
    # Final pred is argmax of class scores (ties -> lower class).
    scores = np.zeros((len(df), 5), dtype=float)

    # core rules
    scores[:, 0] = mx7["VL"]
    scores[:, 1] = mx7["L"]
    scores[:, 2] = mx7["M"]
    scores[:, 3] = mx7["H"]
    scores[:, 4] = mx7["VH"]

    if mode == "overlap10":
        # refinement rules (weighted + clipped)
        def add_rule(target_idx, fire):
            fire = clip01(ref_weight * fire)
            scores[:, target_idx] = np.maximum(scores[:, target_idx], fire)

        # 6) over12 AND S High -> 2
        add_rule(1, np.minimum(g12, mS["High"]))

        # 7) over23 AND S Low -> 2
        add_rule(1, np.minimum(g23, mS["Low"]))

        # 8) over23 AND S High -> 3
        add_rule(2, np.minimum(g23, mS["High"]))

        # 9) over34 AND S High -> 4
        add_rule(3, np.minimum(g34, mS["High"]))

        # 10) over45 AND S High -> 5
        add_rule(4, np.minimum(g45, mS["High"]))

    # crisp (optional) – weighted average of class centers
    centers = np.array([1,2,3,4,5], dtype=float)
    denom = scores.sum(axis=1)
    crisp = np.where(denom > 0, (scores * centers).sum(axis=1) / denom, 3.0)

    pred = np.argmax(scores, axis=1) + 1
    return pred, np.round(crisp, 3)

def confusion_matrix(y_true, y_pred, labels=(1,2,3,4,5)):
    mat = pd.crosstab(pd.Series(y_true, name="true"),
                      pd.Series(y_pred, name="pred"),
                      rownames=["true"], colnames=["pred"])
    # ensure all labels exist
    for l in labels:
        if l not in mat.index: mat.loc[l] = 0
        if l not in mat.columns: mat[l] = 0
    mat = mat.sort_index().sort_index(axis=1)
    return mat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", choices=["core5", "overlap10"], default="overlap10")
    ap.add_argument("--ref_weight", type=float, default=1.0)
    ap.add_argument("--gate_relax", type=float, default=0.25)
    args = ap.parse_args()

    inp = Path(args.input)
    df = pd.read_csv(inp)

    # build S if not present
    if "S" not in df.columns:
        df["S"] = df[["x1","x2","x3","x4","x5","x6"]].mean(axis=1)

    y_true = df["remarks"].astype(int).to_numpy()
    y_pred, crisp = infer_fis3(df, mode=args.mode, ref_weight=args.ref_weight, gate_relax=args.gate_relax)

    acc = (y_pred == y_true).mean()

    print(f"\n=== FIS-3 (x7 + S) mode={args.mode} ===")
    print(f"ref_weight={args.ref_weight}, gate_relax={args.gate_relax}")
    print(f"Input: {inp.resolve()}")
    print(f"Accuracy: {acc:.4f}\n")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):\n")
    print(cm)

    out_dir = Path("results/fis_results/fis-3")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"fis3_x7_S_{args.mode}_w{args.ref_weight}_g{args.gate_relax}"
    cm_path = out_dir / f"confusion_{inp.stem}_{tag}.csv"
    pred_path = out_dir / f"predictions_{inp.stem}_{tag}.csv"

    cm.to_csv(cm_path)

    out = df.copy()
    out["fis3_pred"] = y_pred
    out["fis3_crisp"] = crisp
    out.to_csv(pred_path, index=False)

    print(f"\nSaved confusion matrix to: {cm_path.resolve()}")
    print(f"Saved predictions to: {pred_path.resolve()}")

if __name__ == "__main__":
    main()
