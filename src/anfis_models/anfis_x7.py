#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Utils
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def clip_sigmas(s: np.ndarray, min_sigma: float = 1e-3) -> np.ndarray:
    return np.maximum(s, min_sigma)

def pretty_confusion(cm: np.ndarray, labels: List[int]) -> str:
    # cm rows=true, cols=pred
    w = max(5, max(len(str(x)) for x in labels) + 2)
    head = "true\\pred".ljust(w) + "".join(str(l).rjust(w) for l in labels)
    lines = [head]
    for i, tl in enumerate(labels):
        row = str(tl).ljust(w) + "".join(str(int(cm[i, j])).rjust(w) for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines) + "\n"

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[int(t)], idx[int(p)]] += 1
    return cm

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))

def mse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean((y_true - y_hat) ** 2))

def mae(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_hat)))


# ----------------------------
# MinMax scaler (train-only fit)
# ----------------------------

@dataclass
class MinMax1D:
    x_min: float
    x_max: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = (self.x_max - self.x_min)
        if denom == 0:
            return np.zeros_like(x)
        return (x - self.x_min) / denom

    def inverse(self, x01: np.ndarray) -> np.ndarray:
        return x01 * (self.x_max - self.x_min) + self.x_min


# ----------------------------
# ANFIS 1D (x7 only)
# y = sum_i w_i * (a_i * x + b_i) / sum_i w_i
# w_i = exp(-(x-c_i)^2/(2*s_i^2))
# ----------------------------

def gauss_mf(x: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    # x: (N,), c/s: (M,)
    # returns mu: (N, M)
    x = x.reshape(-1, 1)
    c = c.reshape(1, -1)
    s = s.reshape(1, -1)
    s = np.maximum(s, 1e-6)
    z = (x - c) / s
    return np.exp(-0.5 * (z ** 2))

def anfis_forward(x: np.ndarray, c: np.ndarray, s: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_hat: (N,)
      w:     (N,M) unnormalized firing
      wbar:  (N,M) normalized firing
    """
    w = gauss_mf(x, c, s)                           # (N,M)
    w_sum = np.sum(w, axis=1, keepdims=True) + 1e-12
    wbar = w / w_sum                                # (N,M)
    # rule outputs: f_i(x) = a_i*x + b_i
    f = x.reshape(-1, 1) * a.reshape(1, -1) + b.reshape(1, -1)   # (N,M)
    y_hat = np.sum(wbar * f, axis=1)                # (N,)
    return y_hat, w, wbar

def solve_consequents_ridge(x: np.ndarray, y: np.ndarray, wbar: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for a_i, b_i in:
      y ≈ sum_i wbar_i(x) * (a_i*x + b_i)
    Linear in params.
    """
    N, M = wbar.shape
    # Build design matrix Phi: (N, 2M)
    # For each i: columns are wbar_i*x and wbar_i*1
    Phi = np.zeros((N, 2 * M), dtype=float)
    for i in range(M):
        Phi[:, 2 * i] = wbar[:, i] * x
        Phi[:, 2 * i + 1] = wbar[:, i]

    # Ridge: (Phi^T Phi + λI) theta = Phi^T y
    A = Phi.T @ Phi
    A += ridge * np.eye(A.shape[0])
    rhs = Phi.T @ y
    theta = np.linalg.solve(A, rhs)  # (2M,)

    a = theta[0::2]
    b = theta[1::2]
    return a, b

def init_mfs_quantiles(x_train: np.ndarray,
                       mf_count: int,
                       sigma_mult: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust 1-D Gaussian MF initialisation.

    centres  : q-quantiles  (5 % … 95 %)
    sigmas   : ½ (min dist to neighbours) x sigma_mult / √(2 ln 2)
    """
    # centres
    q = np.linspace(0.05, 0.95, mf_count)
    centres = np.quantile(x_train, q).astype(float)          # (m,)

    # local spacing (min distance to nearest neighbour)
    left  = np.roll(centres, 1)
    right = np.roll(centres, -1)
    # distances to neighbours; use np.inf for edges
    d_left  = centres - left
    d_left[0]   = np.inf
    d_right = right - centres
    d_right[-1] = np.inf
    d_min = np.minimum(d_left, d_right)                      # (m,)

    # convert to σ (≈ half-width / √(2 ln 2))
    sigmas = sigma_mult * d_min / np.sqrt(2 * np.log(2))
    sigmas = clip_sigmas(sigmas, 1e-3)

    return centres, sigmas
    
def optimize_thresholds(y_hat: np.ndarray, y_true: np.ndarray) -> List[float]:
    """
    Find thresholds t1..t4 that maximize accuracy for ordered classes {1..5}.
    We do a fast grid search over candidate thresholds from y_hat quantiles.

    Mapping:
      <=t1 -> 1
      (t1,t2] -> 2
      (t2,t3] -> 3
      (t3,t4] -> 4
      >t4 -> 5
    """
    # candidates from quantiles of predictions
    q = np.linspace(0.05, 0.95, 60)
    cand = np.quantile(y_hat, q)
    cand = np.unique(np.clip(cand, 1.0, 5.0))

    best_acc = -1.0
    best_t = [1.5, 2.5, 3.5, 4.5]

    # brute force but small: ~60^4 is too big, so we do nested with pruning:
    # enforce increasing thresholds and search progressively.
    for t1 in cand:
        for t2 in cand[cand > t1]:
            for t3 in cand[cand > t2]:
                for t4 in cand[cand > t3]:
                    y_pred = apply_thresholds(y_hat, [t1, t2, t3, t4])
                    acc = accuracy(y_true, y_pred)
                    if acc > best_acc:
                        best_acc = acc
                        best_t = [float(t1), float(t2), float(t3), float(t4)]
    return best_t

def apply_thresholds(y_hat: np.ndarray, thresholds: List[float]) -> np.ndarray:
    t1, t2, t3, t4 = thresholds
    y_pred = np.empty_like(y_hat, dtype=int)
    y_pred[y_hat <= t1] = 1
    y_pred[(y_hat > t1) & (y_hat <= t2)] = 2
    y_pred[(y_hat > t2) & (y_hat <= t3)] = 3
    y_pred[(y_hat > t3) & (y_hat <= t4)] = 4
    y_pred[y_hat > t4] = 5
    return y_pred

def round_decode(y_hat: np.ndarray) -> np.ndarray:
    y_pred = np.rint(y_hat).astype(int)
    return np.clip(y_pred, 1, 5)


# ----------------------------
# Training
# ----------------------------

def train_anfis_x7(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    mf_count: int,
    epochs: int,
    lr: float,
    ridge: float,
    sigma_mult: float,
    seed: int,
    thresholds_mode: str,
    patience: Optional[int],
    min_delta: float,
    monitor: str,
) -> Tuple[Dict, Dict, Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      config_like, history, metrics, best_cm, final_cm, best_thresholds, final_thresholds
    """
    rng = np.random.default_rng(seed)

    # init premise
    c, s = init_mfs_quantiles(x_train, mf_count, sigma_mult=sigma_mult)

    # init consequents
    # First forward with dummy a/b so we can solve consequents properly
    a = np.zeros(mf_count, dtype=float)
    b = np.linspace(1.0, 5.0, mf_count, dtype=float)  # sensible start

    best = {
        "epoch": 1,
        "test_mse": float("inf"),
        "test_acc": -1.0,
        "c": c.copy(),
        "s": s.copy(),
        "a": a.copy(),
        "b": b.copy(),
        "thresholds": [1.5, 2.5, 3.5, 4.5],
        "yhat_test": None,
    }

    hist = {
        "epoch": [],
        "train_mse": [],
        "test_mse": [],
        "train_acc": [],
        "test_acc": [],
        "train_mae": [],
        "test_mae": [],
        "thresholds": [],
        "delta_c": [],
        "delta_s": [],
    }

    # early stopping state
    if patience is not None:
        wait = 0
        best_mon = float("inf") if monitor == "mse" else -float("inf")

    for ep in range(1, epochs + 1):
        c_prev = c.copy()
        s_prev = s.copy()

        # forward (current premise)
        yhat_train, w_train, wbar_train = anfis_forward(x_train, c, s, a, b)

        # hybrid step: solve consequents by ridge LS using current wbar
        a, b = solve_consequents_ridge(x_train, y_train, wbar_train, ridge=ridge)

        # forward again with updated consequents
        yhat_train, w_train, wbar_train = anfis_forward(x_train, c, s, a, b)
        yhat_test, w_test, wbar_test = anfis_forward(x_test, c, s, a, b)

        # thresholds
        if thresholds_mode == "optimize":
            th = optimize_thresholds(yhat_train, y_train)
            ypred_train = apply_thresholds(yhat_train, th)
            ypred_test = apply_thresholds(yhat_test, th)
        else:
            th = [1.5, 2.5, 3.5, 4.5]
            ypred_train = round_decode(yhat_train)
            ypred_test = round_decode(yhat_test)

        tr_mse = mse(y_train, yhat_train)
        te_mse = mse(y_test, yhat_test)
        tr_mae = mae(y_train, yhat_train)
        te_mae = mae(y_test, yhat_test)
        tr_acc = accuracy(y_train, ypred_train)
        te_acc = accuracy(y_test, ypred_test)

        # gradient for premise params (c,s) on train MSE
        # y = sum_i wbar_i * (a_i x + b_i)
        # w_i = exp(-0.5 * ((x-c_i)/s_i)^2)
        # dw_i/dc_i = w_i * ((x-c_i)/s_i^2)
        # dw_i/ds_i = w_i * ((x-c_i)^2 / s_i^3)
        #
        # y depends on wbar, which depends on all w's. We'll compute dy/dw using quotient rule.
        # Let S = sum_j w_j
        # wbar_i = w_i / S
        # y = sum_i (w_i/S) * f_i
        # => y = (1/S) * sum_i w_i f_i
        # dy/dw_k = (f_k*S - sum_i w_i f_i) / S^2 = (f_k - y) / S
        #
        # then dy/dc_k = dy/dw_k * dw_k/dc_k, similarly for s_k
        #
        S = np.sum(w_train, axis=1, keepdims=True) + 1e-12          # (N,1)
        f = x_train.reshape(-1, 1) * a.reshape(1, -1) + b.reshape(1, -1)  # (N,M)
        y = yhat_train.reshape(-1, 1)                                # (N,1)
        dy_dw = (f - y) / S                                          # (N,M)

        # error derivative: dMSE/dy = 2*(yhat - ytrue)/N
        dL_dy = (2.0 / x_train.shape[0]) * (yhat_train - y_train)     # (N,)
        dL_dy = dL_dy.reshape(-1, 1)                                  # (N,1)

        xcol = x_train.reshape(-1, 1)                                 # (N,1)
        c_row = c.reshape(1, -1)
        s_row = s.reshape(1, -1)

        # dw/dc and dw/ds
        dw_dc = w_train * ((xcol - c_row) / (s_row ** 2))            # (N,M)
        dw_ds = w_train * (((xcol - c_row) ** 2) / (s_row ** 3))     # (N,M)

        # chain
        dL_dc = np.sum(dL_dy * dy_dw * dw_dc, axis=0)                # (M,)
        dL_ds = np.sum(dL_dy * dy_dw * dw_ds, axis=0)                # (M,)

        # update
        c = c - lr * dL_dc
        s = s - lr * dL_ds
        s = clip_sigmas(s, 1e-3)

        dc = float(np.mean(np.abs(c - c_prev)))
        ds = float(np.mean(np.abs(s - s_prev)))

        # record
        hist["epoch"].append(ep)
        hist["train_mse"].append(tr_mse)
        hist["test_mse"].append(te_mse)
        hist["train_mae"].append(tr_mae)
        hist["test_mae"].append(te_mae)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        hist["thresholds"].append(th)
        hist["delta_c"].append(dc)
        hist["delta_s"].append(ds)

        # best selection: primarily by test_acc, tie-break by test_mse
        is_better = False
        if te_acc > best["test_acc"] + 1e-12:
            is_better = True
        elif abs(te_acc - best["test_acc"]) <= 1e-12 and te_mse < best["test_mse"] - 1e-12:
            is_better = True

        if is_better:
            best.update({
                "epoch": ep,
                "test_mse": te_mse,
                "test_acc": te_acc,
                "c": c.copy(),
                "s": s.copy(),
                "a": a.copy(),
                "b": b.copy(),
                "thresholds": list(map(float, th)),
                "yhat_test": yhat_test.copy(),
            })

        # early stopping
        if patience is not None:
            current_mon = te_mse if monitor == "mse" else te_acc
            improved = False
            if monitor == "mse":
                improved = (best_mon - current_mon) >= min_delta
                if improved:
                    best_mon = current_mon
            else:
                improved = (current_mon - best_mon) >= min_delta
                if improved:
                    best_mon = current_mon

            if improved:
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    # stop
                    break

    # final evaluation at last epoch state
    yhat_train, _, _ = anfis_forward(x_train, c, s, a, b)
    yhat_test, _, _ = anfis_forward(x_test, c, s, a, b)

    if thresholds_mode == "optimize":
        final_th = optimize_thresholds(yhat_train, y_train)
        ypred_test_final = apply_thresholds(yhat_test, final_th)
    else:
        final_th = [1.5, 2.5, 3.5, 4.5]
        ypred_test_final = round_decode(yhat_test)

    ypred_test_best = apply_thresholds(best["yhat_test"], best["thresholds"]) if thresholds_mode == "optimize" else round_decode(best["yhat_test"])

    labels = [1, 2, 3, 4, 5]
    cm_best = confusion_matrix(y_test, ypred_test_best, labels)
    cm_final = confusion_matrix(y_test, ypred_test_final, labels)

    metrics = {
        "best_epoch": {
            "epoch": int(best["epoch"]),
            "test_acc": float(best["test_acc"]),
            "test_mse": float(best["test_mse"]),
            "thresholds": best["thresholds"],
        },
        "final_epoch": {
            "epoch": int(hist["epoch"][-1]),
            "test_acc": float(accuracy(y_test, ypred_test_final)),
            "test_mse": float(mse(y_test, yhat_test)),
            "thresholds": list(map(float, final_th)),
        }
    }

    history = hist

    params = {
        "c": c, "s": s, "a": a, "b": b,
        "best_c": best["c"], "best_s": best["s"], "best_a": best["a"], "best_b": best["b"],
    }
    return params, history, metrics, cm_best, cm_final, np.array(best["thresholds"]), np.array(final_th)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train csv")
    ap.add_argument("--test", required=True, help="test csv")
    ap.add_argument("--out_dir", required=True, help="base output dir")
    ap.add_argument("--run_name", required=True, help="e.g., iter1")
    ap.add_argument("--model_name", required=True, help="e.g., anfis_x7")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.1, help="premise GD lr")
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42111)
    ap.add_argument("--mf_count", type=int, default=5)
    ap.add_argument("--scale", action="store_true", help="MinMax scale x7 using TRAIN only")
    ap.add_argument("--sigma_mult", type=float, default=1.0, help="Multiply initial sigma (wider overlap)")
    ap.add_argument("--thresholds", choices=["round", "optimize"], default="optimize",
                    help="How to map y_hat to classes. optimize usually boosts accuracy.")
    ap.add_argument("--patience", type=int, default=None, help="early stopping patience (optional)")
    ap.add_argument("--min_delta", type=float, default=0.0, help="min improvement for early stopping")
    ap.add_argument("--monitor", choices=["mse", "acc"], default="mse", help="metric to monitor for early stopping")

    args = ap.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # labels must be 1..5 ints
    y_train = train_df["remarks"].to_numpy(dtype=float)
    y_test = test_df["remarks"].to_numpy(dtype=float)

    x_train = train_df["x7"].to_numpy(dtype=float)
    x_test = test_df["x7"].to_numpy(dtype=float)

    scaler = None
    if args.scale:
        scaler = MinMax1D(float(np.min(x_train)), float(np.max(x_train)))
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    out_base = Path(args.out_dir) / args.run_name / args.model_name
    ensure_dir(out_base)

    config = {
        "model": "ANFIS-x7 (1D Gaussian TS-first-order)",
        "train_csv": args.train,
        "test_csv": args.test,
        "epochs": args.epochs,
        "lr": args.lr,
        "ridge": args.ridge,
        "seed": args.seed,
        "mf_count": args.mf_count,
        "scale": bool(args.scale),
        "scaler": asdict(scaler) if scaler is not None else None,
        "sigma_mult": args.sigma_mult,
        "thresholds": args.thresholds,
        "early_stopping": {
            "enabled": args.patience is not None,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "monitor": args.monitor,
        }
    }

    params, history, metrics, cm_best, cm_final, th_best, th_final = train_anfis_x7(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        mf_count=args.mf_count,
        epochs=args.epochs,
        lr=args.lr,
        ridge=args.ridge,
        sigma_mult=args.sigma_mult,
        seed=args.seed,
        thresholds_mode=args.thresholds,
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.monitor,
    )

    # save files
    save_json(out_base / "config.json", config)
    save_json(out_base / "train_history.json", history)
    save_json(out_base / "metrics.json", metrics)

    # confusion matrices
    labels = [1, 2, 3, 4, 5]
    (out_base / "confusion_matrix_best.csv").write_text(pd.DataFrame(cm_best, index=labels, columns=labels).to_csv(), encoding="utf-8")
    (out_base / "confusion_matrix_final.csv").write_text(pd.DataFrame(cm_final, index=labels, columns=labels).to_csv(), encoding="utf-8")
    (out_base / "confusion_matrix_best_pretty.txt").write_text(pretty_confusion(cm_best, labels), encoding="utf-8")
    (out_base / "confusion_matrix_final_pretty.txt").write_text(pretty_confusion(cm_final, labels), encoding="utf-8")

    # print summary
    be = metrics["best_epoch"]
    fe = metrics["final_epoch"]
    print("Done.")
    print("Saved to:", str(out_base))
    print(f"Best epoch: {be['epoch']} | best acc={be['test_acc']:.4f} mse={be['test_mse']:.6f} thresholds={be['thresholds']}")
    print(f"Final epoch: {fe['epoch']} | final acc={fe['test_acc']:.4f} mse={fe['test_mse']:.6f} thresholds={fe['thresholds']}")
    print(f"Last update drift: Δc={history['delta_c'][-1]:.3e} Δs={history['delta_s'][-1]:.3e} (should NOT be 0.0)")

if __name__ == "__main__":
    main()
