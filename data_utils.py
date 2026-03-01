"""
data_utils.py
Download UCI Heart Disease dataset, preprocess it, and partition it
into non-IID splits to simulate heterogeneous rural/urban clinics.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import os


# ─────────────────────────────────────────────────────────────────────────────
# Dataset download + preprocessing
# ─────────────────────────────────────────────────────────────────────────────

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_heart_disease_data(data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Download and preprocess the UCI Heart Disease dataset.
    Returns (X, y) as numpy arrays, ready for splitting.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "heart.csv")

    if not os.path.exists(cache_path):
        print("Downloading UCI Heart Disease dataset...")
        import urllib.request
        try:
            urllib.request.urlretrieve(UCI_URL, cache_path)
        except Exception:
            # Fallback: generate synthetic data with same structure
            print("⚠️  Download failed — generating synthetic data with same structure.")
            return _generate_synthetic_data()

    df = pd.read_csv(cache_path, header=None, names=FEATURE_NAMES + ["target"])
    df.replace("?", np.nan, inplace=True)
    df = df.astype(float)
    df.dropna(inplace=True)

    # Binary classification: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{y.mean()*100:.1f}% positive class")
    return X, y


def _generate_synthetic_data(n_samples: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic heart disease data when UCI download fails.
    Preserves same structure: 13 features, binary target.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, 13).astype(np.float32)
    # Make it somewhat linearly separable
    weights = np.random.randn(13)
    logits = X @ weights
    y = (logits > 0).astype(np.int64)
    print(f"Synthetic data generated: {n_samples} samples")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Non-IID partitioning to simulate heterogeneous clinics
# ─────────────────────────────────────────────────────────────────────────────

def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    num_clinics: int = 5,
    partition_strategy: str = "non_iid",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple], Tuple]:
    """
    Split data into per-clinic training sets and a shared global test set.

    Strategies:
        "iid"     — Random shuffle, equal split (unrealistic baseline)
        "non_iid" — Sorted by class, unequal splits (realistic rural scenario)

    Returns:
        clinic_datasets: List of (X_train, y_train) per clinic
        test_data:       (X_test, y_test) global evaluation set
    """
    np.random.seed(seed)

    # Hold out global test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    if partition_strategy == "non_iid":
        clinic_datasets = _non_iid_partition(X_train, y_train, num_clinics)
    else:
        clinic_datasets = _iid_partition(X_train, y_train, num_clinics)

    # Print partition summary
    print(f"\n{'='*55}")
    print(f"  Data Partitioning: {partition_strategy.upper()} | {num_clinics} clinics")
    print(f"{'='*55}")
    for i, (Xc, yc) in enumerate(clinic_datasets):
        clinic_type = "🏙️  Urban" if i == 0 else "🏥 Rural "
        pos_rate = yc.mean() * 100
        print(f"  Clinic {i+1} ({clinic_type}): {len(yc):4d} samples | "
              f"{pos_rate:.0f}% positive")
    print(f"  Global test set:           {len(y_test):4d} samples")
    print(f"{'='*55}\n")

    return clinic_datasets, (X_test, y_test)


def _iid_partition(X, y, num_clinics):
    idx = np.random.permutation(len(y))
    splits = np.array_split(idx, num_clinics)
    return [(X[s], y[s]) for s in splits]


def _non_iid_partition(X, y, num_clinics):
    """
    Simulate realistic clinic heterogeneity:
    - Clinic 0 (urban): large dataset, balanced classes
    - Clinics 1-N (rural): small datasets, skewed class distributions
    """
    # Sort by label to create natural class imbalance
    sorted_idx = np.argsort(y)
    n = len(sorted_idx)

    # Unequal splits: urban clinic gets ~40%, rural clinics split the rest
    urban_size = int(n * 0.40)
    rural_total = n - urban_size
    rural_per_clinic = rural_total // (num_clinics - 1)

    clinic_datasets = []

    # Urban clinic — large, balanced
    urban_idx = sorted_idx[:urban_size]
    np.random.shuffle(urban_idx)
    clinic_datasets.append((X[urban_idx], y[urban_idx]))

    # Rural clinics — small, skewed
    for i in range(num_clinics - 1):
        start = urban_size + i * rural_per_clinic
        end = start + rural_per_clinic if i < num_clinics - 2 else n
        rural_idx = sorted_idx[start:end]
        np.random.shuffle(rural_idx)
        clinic_datasets.append((X[rural_idx], y[rural_idx]))

    return clinic_datasets


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset + DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class ClinicDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 16,
                    shuffle: bool = True) -> DataLoader:
    dataset = ClinicDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
