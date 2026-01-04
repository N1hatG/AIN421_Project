#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Takagi–Sugeno ANFIS (first-order consequents, Gaussian MFs)
• unlimited number of input columns
• arbitrary #MFs per input
• ridge-LS for consequents  +  SGD for premise (centres/sigmas)
• optional one-shot threshold optimiser (epoch 3) for 5-class decoding
Tested: 7-D model (2,2,2,2,2,2,3) ⇒ 192 rules on 37 500 × 7 samples.
"""


import argparse, itertools, json, math
from dataclasses import asdict, dataclass
from pathlib   import Path
from typing    import Dict, List, Tuple, Optional

import time 

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None: p.mkdir(parents=True, exist_ok=True)
def save_json(path: Path, obj):  path.write_text(json.dumps(obj, indent=2))

def clip_sigmas(s: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    return np.maximum(s, eps)

def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y == y_pred))

def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean((y - y_hat) ** 2))

def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y_hat)))

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: List[int]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    cm  = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        cm[idx[int(t)], idx[int(p)]] += 1
    return cm

def pretty_cm(cm: np.ndarray, labels: List[int]) -> str:
    w = max(5, max(len(str(x)) for x in labels) + 2)
    head = "true\\pred".ljust(w) + "".join(str(l).rjust(w) for l in labels)
    rows = [head]
    for i, tl in enumerate(labels):
        rows.append(str(tl).ljust(w) +
                    "".join(str(cm[i, j]).rjust(w) for j in range(len(labels))))
    return "\n".join(rows) + "\n"

@dataclass
class MinMax1D:
    lo: float
    hi: float
    def transform(self, x): return (x - self.lo) / (self.hi - self.lo + 1e-12)
    def inv(self, x01):     return x01 * (self.hi - self.lo) + self.lo

def init_mfs_quantiles(x: np.ndarray,
                       mf_counts: List[int],
                       sigma_mult: float = 1.0
                      ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    x shape (N,D) – returns lists c_k  s_k  (each of shape (mf_k,))
    """
    c_list, s_list = [], []
    for col, m in enumerate(mf_counts):
        xi = x[:, col]
        qs = np.linspace(0.1, 0.9, m)
        c  = np.quantile(xi, qs).astype(float)
        # sigma: average spacing
        spacing = np.diff(np.sort(c))
        base = float(spacing.mean()) if spacing.size else float(np.std(xi))
        if base < 1e-6: base = 1.0
        s = np.ones_like(c) * base * sigma_mult
        c_list.append(c)
        s_list.append(clip_sigmas(s))
    return c_list, s_list

def apply_thr(y_hat: np.ndarray, thr: List[float]) -> np.ndarray:
    t1,t2,t3,t4 = thr
    y = np.empty_like(y_hat, int)
    y[y_hat <= t1]                          = 1
    y[(y_hat>t1)&(y_hat<=t2)]               = 2
    y[(y_hat>t2)&(y_hat<=t3)]               = 3
    y[(y_hat>t3)&(y_hat<=t4)]               = 4
    y[y_hat > t4]                           = 5
    return y

def optimise_thr(y_hat: np.ndarray, y_true: np.ndarray) -> List[float]:
    cand  = np.unique(np.quantile(y_hat, np.linspace(.05,.95,60)))
    best, best_thr = -1.0, [1.5,2.5,3.5,4.5]
    for t1 in cand:
      for t2 in cand[cand>t1]:
        for t3 in cand[cand>t2]:
          for t4 in cand[cand>t3]:
            acc = accuracy(y_true, apply_thr(y_hat,[t1,t2,t3,t4]))
            if acc > best:
               best, best_thr = acc, [float(t1),float(t2),float(t3),float(t4)]
    return best_thr

round_thr = [1.5,2.5,3.5,4.5]


def enumerate_rules(mf_counts: List[int]) -> np.ndarray:
    """
    Returns array shape (M,D) each row the MF-index per input.
    """
    combos = list(itertools.product(*[range(m) for m in mf_counts]))
    return np.array(combos, int)      # (M,D)

def gauss_mu(x_col: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    x_col (N,) ,  c/s (mf_k,)  → μ (N,mf_k)
    """
    z = (x_col[:, None] - c[None, :]) / s[None, :]
    return np.exp(-0.5 * z * z)

def forward(X: np.ndarray,
            c_list: List[np.ndarray],
            s_list: List[np.ndarray],
            a: np.ndarray,   # (M,D)
            b: np.ndarray,   # (M,)
            rule_tbl: np.ndarray  # (M,D)
            ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,List[np.ndarray]]:
    """
    Return y_hat (N,),  w  (N,M),  wbar (N,M),  μ_list per input
    """
    N, D = X.shape
    mu_list = [gauss_mu(X[:,k], c_list[k], s_list[k]) for k in range(D)]
    # gather membership for each rule
    w = np.ones((N, rule_tbl.shape[0]))
    for k in range(D):
        w *= mu_list[k][:, rule_tbl[:,k]]          # broadcast
    w_sum = w.sum(1, keepdims=True) + 1e-12
    wbar  = w / w_sum
    f     = X @ a.T + b            # (N,M)
    y_hat = (wbar * f).sum(1)
    return y_hat, w, wbar, mu_list

def solve_consequents(X: np.ndarray, y: np.ndarray,
                      wbar: np.ndarray, ridge: float,
                      D: int, M:int) -> Tuple[np.ndarray,np.ndarray]:
    """
    returns a (M,D)  slopes per input,  b (M,)
    """
    N = X.shape[0]
    Phi = np.zeros((N, M*(D+1)))
    offset = 0
    ones = np.ones((N,1))
    for i in range(M):
        Phi[:, offset:offset+D]     = wbar[:,i:i+1] * X
        Phi[:, offset+D]            = wbar[:,i]      
        offset += D+1
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    theta = np.linalg.solve(A, Phi.T @ y)            # (M*(D+1),)
    a = theta.reshape(M, D+1)[:,:D]
    b = theta.reshape(M, D+1)[:,-1]
    return a, b

def train_generic(X_tr, y_tr, X_te, y_te,
                  mf_counts: List[int], epochs:int, lr:float, ridge:float,
                  sigma_mult:float, seed:int,
                  thr_mode:str, patience:int|None, min_delta:float, monitor:str,thr_refresh):
    rng = np.random.default_rng(seed)
    N,D = X_tr.shape
    c_list, s_list = init_mfs_quantiles(X_tr, mf_counts, sigma_mult)
    rule_tbl = enumerate_rules(mf_counts)           # (M,D)
    M = rule_tbl.shape[0]

    a = np.zeros((M,D))
    b = np.linspace(1.0,5.0,M)


    history = {k:[] for k in
       ["epoch","train_mse","test_mse","train_acc","test_acc",
        "train_mae","test_mae","delta_c","delta_s"]}
    best = dict(epoch=1, acc=-1.0, mse=np.inf,
                c=[ci.copy() for ci in c_list], s=[si.copy() for si in s_list],
                a=a.copy(), b=b.copy(), thr=round_thr)

    # early-stopping state
    if patience is not None:
        wait = 0
        best_mon = np.inf if monitor=="mse" else -np.inf

    fixed_thr = round_thr

    for ep in range(1, epochs+1):
        if thr_mode=="optimize" and (ep == 3 or (thr_refresh and ep % thr_refresh == 0)):
            y_hat_tmp, *_ = forward(X_tr, c_list, s_list, a, b, rule_tbl)
            fixed_thr = optimise_thr(y_hat_tmp, y_tr)


        y_hat_tr, w_tr, wbar_tr, mu_list = forward(X_tr,c_list,s_list,a,b,rule_tbl)
     
        a, b = solve_consequents(X_tr, y_tr, wbar_tr, ridge, D, M)
   
        y_hat_tr, w_tr, wbar_tr, mu_list = forward(X_tr,c_list,s_list,a,b,rule_tbl)
        y_hat_te, _, _, _                = forward(X_te,c_list,s_list,a,b,rule_tbl)

        ypred_tr = apply_thr(y_hat_tr, fixed_thr)
        ypred_te = apply_thr(y_hat_te, fixed_thr)

        tr_mse, te_mse = mse(y_tr,y_hat_tr), mse(y_te,y_hat_te)
        tr_mae, te_mae = mae(y_tr,y_hat_tr), mae(y_te,y_hat_te)
        tr_acc, te_acc = accuracy(y_tr,ypred_tr), accuracy(y_te,ypred_te)


        S   = w_tr.sum(1,keepdims=True)+1e-12
        f   = (X_tr @ a.T) + b
        y_  = y_hat_tr[:,None]
        dy_dw = (f - y_) / S
        dL_dy = (2.0/N)*(y_hat_tr - y_tr)[:,None]

        g_rule = dL_dy * dy_dw                     # (N,M)
        g_rule *= w_tr                            

        delta_c = [np.zeros_like(c) for c in c_list]
        delta_s = [np.zeros_like(s) for s in s_list]

        for m in range(M):
            for k in range(D):
                j = rule_tbl[m,k]                  # MF index for variable k
                xk = X_tr[:,k]
                ck, sk = c_list[k][j], s_list[k][j]
    
                fac_c = (xk - ck)  / (sk**2)
                fac_s = ((xk - ck)**2) / (sk**3)
                delta_c[k][j] += np.sum(g_rule[:,m] * fac_c)
                delta_s[k][j] += np.sum(g_rule[:,m] * fac_s)

        c_prev_all = [ci.copy() for ci in c_list]
        s_prev_all = [si.copy() for si in s_list]

        for k in range(D):
            c_list[k] -= lr * delta_c[k]
            s_list[k] -= lr * delta_s[k]
            s_list[k]  = clip_sigmas(s_list[k])
        dc = np.mean([np.mean(np.abs(c_list[k] - c_prev_all[k])) for k in range(D)])
        ds = np.mean([np.mean(np.abs(s_list[k] - s_prev_all[k])) for k in range(D)])
        
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"ep {ep:3d}/{epochs} │ "
                  f"test_acc {te_acc:5.3f} │ "
                  f"test_mse {te_mse:7.4f} │ "
                  f"Δc {dc:6.2e} Δs {ds:6.2e}",
                  flush=True)

        for k,v in [("epoch",ep),("train_mse",tr_mse),("test_mse",te_mse),
                    ("train_acc",tr_acc),("test_acc",te_acc),
                    ("train_mae",tr_mae),("test_mae",te_mae),
                    ("delta_c",dc),("delta_s",ds)]: history[k].append(v)

        better = (te_acc > best['acc']+1e-12) or \
                 (abs(te_acc-best['acc'])<=1e-12 and te_mse<best['mse']-1e-12)
        if better:
            best.update(epoch=ep,acc=te_acc,mse=te_mse,thr=fixed_thr.copy(),
                        c=[ci.copy() for ci in c_list], s=[si.copy() for si in s_list],
                        a=a.copy(), b=b.copy())

        if patience is not None:
            cur = te_mse if monitor=="mse" else te_acc
            improve = (best_mon-cur>=min_delta) if monitor=="mse" else (cur-best_mon>=min_delta)
            if improve: best_mon,wait = cur,0
            else:
                wait += 1
                if wait>=patience: break

    return history,best,c_list,s_list,a,b,rule_tbl


def main():
    p = argparse.ArgumentParser(description="Generic multi-input ANFIS trainer")
    p.add_argument("--train", required=True)
    p.add_argument("--test",  required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_name", required=True)

    p.add_argument("--input_cols", nargs="+",
                   default=["x1","x2","x3","x4","x5","x6","x7"])
    p.add_argument("--mf_counts",  nargs="+", type=int,
                   default=[2,2,2,2,2,2,3])

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr",     type=float, default=0.003)
    p.add_argument("--ridge",  type=float, default=1e-4)
    p.add_argument("--sigma_mult", type=float, default=2.0)
    p.add_argument("--seed",   type=int, default=42111)

    p.add_argument("--scale", action="store_true")
    p.add_argument("--thresholds", choices=["round","optimize"], default="optimize")
    p.add_argument("--patience", type=int)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--monitor", choices=["mse","acc"], default="mse")

    p.add_argument("--thr_refresh", type=int, default=5,
               help="re-optimise thresholds every N epochs (0 = only once)")
    
    args = p.parse_args()


    df_tr = pd.read_csv(args.train); df_te = pd.read_csv(args.test)
    X_tr  = df_tr[args.input_cols].to_numpy(float)
    X_te  = df_te[args.input_cols].to_numpy(float)
    y_tr  = df_tr["remarks"].to_numpy(float)
    y_te  = df_te["remarks"].to_numpy(float)

    scaler = None
    if args.scale:
        lo = X_tr.min(0); hi = X_tr.max(0)
        scaler = (lo, hi)
        X_tr = (X_tr-lo)/(hi-lo+1e-12)
        X_te = (X_te-lo)/(hi-lo+1e-12)

    hist,best,c_list,s_list,a,b,rule_tbl = train_generic(
        X_tr,y_tr,X_te,y_te,
        mf_counts = args.mf_counts,
        epochs    = args.epochs,
        lr=args.lr,ridge=args.ridge,sigma_mult=args.sigma_mult,seed=args.seed,
        thr_mode=args.thresholds,
        patience=args.patience,min_delta=args.min_delta,monitor=args.monitor, thr_refresh=args.thr_refresh)


    out_base = Path(args.out_dir)/args.run_name/args.model_name
    ensure_dir(out_base)

    cfg = vars(args)|{"rule_count":int(np.prod(args.mf_counts)),
                      "scaler":None if scaler is None else {"lo":scaler[0].tolist(),
                                                           "hi":scaler[1].tolist()}}
    save_json(out_base/"config.json", cfg)
    save_json(out_base/"train_history.json", hist)

    y_hat_best,_ ,_,_ = forward(X_te,best['c'],best['s'],best['a'],best['b'],rule_tbl)
    y_pred_best = apply_thr(y_hat_best,best['thr'])
    cm_best     = confusion_matrix(y_te,y_pred_best,[1,2,3,4,5])
    (out_base/"confusion_matrix_best_pretty.txt").write_text(pretty_cm(cm_best,[1,2,3,4,5]))

    metrics = {"best_epoch":dict(epoch=best['epoch'],accuracy=best['acc'],mse=best['mse'],
                                 thresholds=best['thr']),
               "final_epoch":dict(epoch=hist['epoch'][-1],accuracy=hist['test_acc'][-1],
                                  mse=hist['test_mse'][-1])}
    save_json(out_base/"metrics.json", metrics)

    print("Saved to",out_base)
    print("Best epoch",best['epoch'],"acc",best['acc'])

if __name__ == "__main__":
    main()
