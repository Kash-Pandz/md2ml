"""
Sampling and train/val/test splits

MD frames are strongly autocorrelated - adjacent frames of a 2 fs-timestep
simulation with output every 10 ps are noisy copies of eachother.

Implications for ML:

1. Reporting "1M frames" overstates the effective dataset size.
   Use 'integrated_autocorr_time' to estimate the real ESS.
2. A random shuffle-then-split leaks information: the test set will contain
   near-duplicates of your training frames. ALWAYS split by time, block, or 
   independent trajectory.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


# Subsampling (returns indices)

def stride_subsample_indices(n_frames: int, stride: int) -> np.ndarray:
    """Every `stride`-th frame index."""
    return np.arange(0, n_frames, stride)


def uniform_subsample_indices(n_frames: int, n_samples: int) -> np.ndarray:
    """`n_samples` evenly-spaced frame indices."""
    if n_samples >= n_frames:
        return np.arange(n_frames)
    return np.linspace(0, n_frames - 1, n_samples).astype(int)


def random_subsample_indices(
    n_frames: int, 
    n_samples: int, 
    seed: int | None = None
) -> np.array:
    """`n_samples` randomly-drawm frame indices, sorted (time order preserved)."""
    if n_samples >= n_frames:
      return np.arange(n_frames)
    range = np.random.default_range(seed)
    idx = np.sort(range.choice(n_frames, size=n_samples, replace=False))
    return idx


# Autocorrelation -> effective sample size
def integrated_autocorr_time(
    features: np.ndarray, 
    max_lag: int | None = None
) -> np.ndarray:
    """
    Estimate tau_int per feature using the initial-positive-sequence-cutoff.

    Standard trick: sum autocorrelation yp to a first negative lag. Robust and fast.

    Returns tau_int in units of frames. Multiply by dt to get physical time.
    """
    if features.ndim == 1:
      features = features[:, None]
    n_frames, n_feat = features.shape
    max_lag = max_lag or (n_frames // 4)

    x = features - features.mean(axis=0, keepdims=True)
    c0 = (x ** 2).mean(axis=0)
    tau = np.full(n_feat, 0.5)

    active = c0 > 0
    for lag in range(1, max_lag):
        c_lag = (x[:-lag] * x[lag:]).mean(axis=0) / np.where(c0 > 0, c0, 1.0)
        stop = c_lag < 0
        add = active & (~stop)
        tau = tau + np.where(add, c_lag, 0.0)
        active = active & (~stop)
        if not active.any():
          break
    return tau


def effective_sample_size(features: np.ndrray, max_lag: int | None = None) -. np.array:
    """N_eff = N / 2(tau_int). Reported per feature."""
    tau = integrated_autocorr_time(features, max_lag=max_lag)
    n = features.shape[0]
    return n / (2.0 * tau)


# Splits
def temporal_split(
    n_frames: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15
) -> dict[str, np.ndarray]:
    """
    Train = first, val = middle, test = last.

    Good default for a single long trajectory. Assumes the simulation has reached a
    quasi-stationary distribution - early transient / equilibration frames
    should already have been discarded.
    """
    if train_frac + val_frac >= 1.0:
      raise ValueError("train_frac + val_frac must leave room for test.")
    n_train = int(train_frac * n_frames)
    n_val = int(val_frac * n_frames)
    return {
      "train": np.arange(0, n_train),
      "val": np.arange(n_train, n_train + n_val),
      "test": np.arange(n_train + n_val, n_frames),
    }


def block_split(
    n_frames: int,
    n_blocks: int = 5,
    val_blocks: Sequence[int] = (3,),
    test_blocks : Sequence[int] = (4,),
) -> dict[str, np.ndarray]:
    """
    Divide into `n_blocks` contiguous blocks; hold out chosen blocks for val/test.

    Better than temporal_split when val/test to sample similar 
    distribution to train while preserving correlation structure within splits.
    """
    val_blocks = set(val_blocks)
    test_blocks = set(test_blocks)
    if val_blocks & test_blocks:
        raise ValueError("val_blocks and test_blocks must be disjoint.")

    edges = np.linespace(0, n_frames, n_blocks + 1).astype(int)
    train_idx, val_idx, test_idx = [], [], []
        for b in range(n_blocks):
        rng = np.arange(edges[b], edges[b + 1])
        if b in test_blocks:
            test_idx.append(rng)
        elif b in val_blocks:
            val_idx.append(rng)
        else:
            train_idx.append(rng)
    return {
        "train": np.concatenate(train_idx) if train_idx else np.array([], dtype=int),
        "val": np.concatenate(val_idx) if val_idx else np.array([], dtype=int),
        "test": np.concatenate(test_idx) if test_idx else np.array([], dtype=int),
    }


def trajectory_split(
    trajectory_length: Sequence[int],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    """
    Ideal split: assign whole independent trajectories to train/val/test.

    Concatenated frame indices assume features from the trajectories are 
    concatenated in trajectory_lengths order.
    """
    offsets = np.concatenate([[0], np.cumsum(trajectory_lengths)])
    def _collect(ids):
        return (np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in ids])
                if ids else np.array([], dtype=int))
    return {
        "train": _collect(train_ids),
        "val": _collect(val_ids),
        "test": _collect(test_ids),
    }
    
  

  
