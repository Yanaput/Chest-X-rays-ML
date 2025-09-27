from pathlib import Path
import json
import numpy as np
from sklearn.metrics import f1_score


def tune_thresholds(
    probs_val: np.ndarray, 
    y_val: np.ndarray,   
    grid=np.linspace(0.05, 0.95, 19)
) -> np.ndarray:
    C = probs_val.shape[1]
    ths = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        p = probs_val[:, c]
        y = y_val[:, c].astype(bool)
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            yhat = p >= t
            tp = np.logical_and(yhat, y).sum()
            fp = np.logical_and(yhat, ~y).sum()
            fn = np.logical_and(~yhat, y).sum()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2*prec*rec/(prec+rec+1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        ths[c] = best_t
    return ths