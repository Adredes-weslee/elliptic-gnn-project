import torch
import numpy as np
from src.utils.metrics import pr_auc_illicit, precision_at_k, pick_threshold_max_f1
from src.data.dataset_elliptic import make_temporal_masks
from torch_geometric.data import Data


def test_temporal_masks():
    x = torch.randn(5, 3)
    y = torch.tensor([0, 1, 0, 1, 0])
    t = torch.tensor([1, 1, 2, 3, 4])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    d = Data(x=x, edge_index=edge_index, y=y)
    d.timestep = t
    d = make_temporal_masks(d, t_train_end=1, t_val_end=3)
    assert d.train_mask.tolist() == [True, True, False, False, False]
    assert d.val_mask.tolist() == [False, False, True, True, False]
    assert d.test_mask.tolist() == [False, False, False, False, True]


def test_metrics_shapes():
    y = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    s = np.linspace(0, 1, len(y))
    auc = pr_auc_illicit(y, s)
    assert 0 <= auc <= 1
    thr, f1 = pick_threshold_max_f1(y, s)
    p_at_3 = precision_at_k(y, s, 3)
    assert 0 <= p_at_3 <= 1
