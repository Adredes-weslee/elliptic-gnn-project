
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class TemperatureScaler(nn.Module):
    # Learn a single temperature on validation logits (binary or multi-class).
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

    def fit(self, logits_val: torch.Tensor, labels_val: torch.Tensor, max_iter: int = 1000):
        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.T], lr=0.1, max_iter=max_iter)

        logits_val = logits_val.detach()
        labels_val = labels_val.detach()

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits_val), labels_val.long())
            loss.backward()
            return loss
        opt.step(closure)
        return float(self.T.detach().cpu().item())

def calibrate_isotonic(y_score_val: np.ndarray, y_true_val: np.ndarray):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_score_val, y_true_val)
    return ir

def calibrate_platt(y_score_val: np.ndarray, y_true_val: np.ndarray):
    # Platt scaling via logistic regression on scores
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(y_score_val.reshape(-1, 1), y_true_val)
    return lr

def apply_sklearn_calibrator(cal, y_score: np.ndarray) -> np.ndarray:
    if hasattr(cal, "predict_proba"):
        return cal.predict_proba(y_score.reshape(-1, 1))[:, 1]
    else:
        return cal.transform(y_score)
