#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANFIS-(S + x7)
2 inputs  :  S = mean(x1..x6) , x7
MF counts : 3 (Low/Med/High)  , 5 (VL..VH)  ⇒ 15 rules
Re-uses train_generic() from anfis_generic.py
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

from anfis_generic import train_generic, apply_thr, enumerate_rules, ensure_dir, save_json

DEF_INPUTS   = ["S", "x7"]
DEF_MF       = [3, 5]
DEF_EPOCHS   = 80
DEF_LR       = 0.01
DEF_RIDGE    = 1e-4
DEF_SIGMULT  = 1.5
DEF_SEED     = 42111


def add_S_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    if "S" not in df.columns:
        df = df.copy()
        df["S"] = df[[f"x{i}" for i in range(1,7)]].mean(axis=1)
    return df

def main() -> None:
    ap = argparse.ArgumentParser("Train ANFIS on (S , x7)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--model_name", default="anfis_Sx7")


    ap.add_argument("--epochs", type=int,   default=DEF_EPOCHS)
    ap.add_argument("--lr",     type=float, default=DEF_LR)
    ap.add_argument("--ridge",  type=float, default=DEF_RIDGE)
    ap.add_argument("--sigma_mult", type=float, default=DEF_SIGMULT)
    ap.add_argument("--seed",   type=int,   default=DEF_SEED)

    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--thresholds", choices=["round","optimize"], default="optimize")
    ap.add_argument("--patience", type=int)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--monitor", choices=["mse","acc"], default="mse")
    args = ap.parse_args()


    df_tr = add_S_if_missing(pd.read_csv(args.train))
    df_te = add_S_if_missing(pd.read_csv(args.test))

    X_tr = df_tr[DEF_INPUTS].to_numpy(float)
    X_te = df_te[DEF_INPUTS].to_numpy(float)
    y_tr = df_tr["remarks"].to_numpy(float)
    y_te = df_te["remarks"].to_numpy(float)

    scaler = None
    if args.scale:
        lo, hi = X_tr.min(0), X_tr.max(0)
        scaler = dict(lo=lo.tolist(), hi=hi.tolist())
        X_tr = (X_tr - lo) / (hi - lo + 1e-12)
        X_te = (X_te - lo) / (hi - lo + 1e-12)


    hist, best, c_list, s_list, a, b, rule_tbl = train_generic(
        X_tr, y_tr, X_te, y_te,
        mf_counts  = DEF_MF,
        epochs     = args.epochs,
        lr         = args.lr,
        ridge      = args.ridge,
        sigma_mult = args.sigma_mult,
        seed       = args.seed,
        thr_mode   = args.thresholds,
        patience   = args.patience,
        min_delta  = args.min_delta,
        monitor    = args.monitor
    )

    out_base = Path(args.out_dir)/args.run_name/args.model_name
    ensure_dir(out_base)

    cfg = vars(args) | {
        "input_cols" : DEF_INPUTS,
        "mf_counts"  : DEF_MF,
        "rule_count" : int(np.prod(DEF_MF)),
        "scaler"     : scaler
    }
    save_json(out_base/"config.json", cfg)
    save_json(out_base/"train_history.json", hist)

    # confusion matrix & metrics
    from anfis_generic import confusion_matrix, pretty_cm
    y_hat_best, *_ = train_generic.forward(X_te, best["c"], best["s"],
                                           best["a"],  best["b"], rule_tbl)
    y_pred_best = apply_thr(y_hat_best, best["thr"])
    cm_best     = confusion_matrix(y_te, y_pred_best, [1,2,3,4,5])
    (out_base/"confusion_matrix_best_pretty.txt").write_text(
        pretty_cm(cm_best,[1,2,3,4,5]))

    metrics = {"best_epoch":dict(epoch=best["epoch"],
                                 accuracy=best["acc"],
                                 mse=best["mse"],
                                 thresholds=best["thr"])}
    save_json(out_base/"metrics.json", metrics)

    print("\nSaved results to", out_base)
    print(f"Best epoch {best['epoch']}  |  test-ACC = {best['acc']:.4f}")

if __name__ == "__main__":
    main()
