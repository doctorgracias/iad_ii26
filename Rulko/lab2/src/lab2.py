import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn

# Global Configuration
SEED = 42
PERPLEXITIES = (20, 40, 60)
CONFIGS = [
    ((16,), 0.001),
    ((16,), 0.003),
    ((32, 16), 0.001),
    ((32, 16), 0.003),
]


# Data Preparation & Math Utilities
def standardize_data(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes standardization statistics strictly on the provided dataset."""
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=1)
    if not np.isfinite(x).all() or np.any(std <= 0):
        raise ValueError("Inputs must contain finite values and non-constant features.")
    return (x - mean) / std, mean, std


def calculate_manual_pca(z: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Custom PCA via eigendecomposition matching numpy.linalg.eig."""
    mean = z.mean(axis=0)
    centered = z - mean
    cov = centered.T @ centered / (len(z) - 1)
    values, vectors = np.linalg.eig(cov)
    values, vectors = np.real_if_close(values), np.real_if_close(vectors)
    
    if np.iscomplexobj(values) or np.iscomplexobj(vectors):
        raise ArithmeticError("Complex matrix decomposition output encountered.")
        
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    
    if values.min() < -1e-9:
        raise ArithmeticError("Negative covariance eigenvalue detected.")
        
    values = np.maximum(values, 0)
    basis = vectors[:, :k]
    signs = np.sign(basis[np.argmax(abs(basis), axis=0), np.arange(k)])
    basis = basis * signs
    scores = centered @ basis
    
    if not np.allclose(basis.T @ basis, np.eye(k), atol=1e-9):
        raise ArithmeticError("PCA basis vectors are not orthonormal.")
        
    return scores, basis, values, mean


def calculate_mse(x: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((x - reconstructed) ** 2))


def get_split_indices(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, holdout = train_test_split(np.arange(n), test_size=0.4, random_state=SEED)
    valid, test = train_test_split(holdout, test_size=0.5, random_state=SEED)
    return train, valid, test


# Model Architecture
class Encoder(nn.Module):
    def __init__(self, input_dim: int, k: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers = []
        widths = [input_dim, *hidden, k]
        for i, (in_dim, out_dim) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, k: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers = []
        widths = [k, *reversed(hidden), input_dim]
        for i, (in_dim, out_dim) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(widths) - 2:
                layers.append