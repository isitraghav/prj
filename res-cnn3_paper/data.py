"""ELPV data loading, stratified split, labels and sample weights.

Reproduces the labeling of Deitsch et al. 2019 (Solar Energy 185:455-468):
per-cell defect probability p in {0.0, 0.33, 0.67, 1.0}. Binary target for
classification/evaluation is defective = p > 0.5 (paper thresholds the CNN
output at 50%, Fig. 9). Sample weights follow the rater-confidence scheme of
Table 2 (confident labels weight 1.0, non-confident 0.33 / 0.67), optionally
combined with the class-balancing weight c_j = S / (2 n_j), Eq. (5).
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image

# Table 2: sample weight from the rater's stated confidence, keyed by p.
CONFIDENCE_WEIGHT = {0.0: 1.0, 0.3333333333333333: 0.33, 0.6666666666666666: 0.67, 1.0: 1.0}


def _nearest_prob(p: float) -> float:
    return min(CONFIDENCE_WEIGHT, key=lambda q: abs(q - p))


def load_labels(root: str):
    """Parse labels.csv -> (rel_paths, probs float, types 'mono'/'poly')."""
    csv = os.path.join(root, "labels.csv")
    paths, probs, types = [], [], []
    with open(csv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, prob, typ = line.split()
            paths.append(path)
            probs.append(float(prob))
            types.append(typ)
    return paths, np.array(probs, dtype=np.float32), np.array(types)


def load_images(root: str, paths, size: int = 300) -> np.ndarray:
    """Load cells as (N, size, size) uint8 grayscale, resized to size x size."""
    out = np.empty((len(paths), size, size), dtype=np.uint8)
    for i, rel in enumerate(paths):
        img = Image.open(os.path.join(root, rel)).convert("L")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(img, dtype=np.uint8)
    return out


def binary_labels(probs: np.ndarray) -> np.ndarray:
    """Defective (1) if p > 0.5, else functional (0). Paper eval threshold."""
    return (probs > 0.5).astype(np.int64)


def confidence_weights(probs: np.ndarray) -> np.ndarray:
    return np.array([CONFIDENCE_WEIGHT[_nearest_prob(p)] for p in probs], dtype=np.float32)


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    """c_j = S / (2 n_j), Eq. (5): inverse class frequency, per sample."""
    S = len(y)
    w = np.empty(S, dtype=np.float32)
    for j in (0, 1):
        n_j = max(int((y == j).sum()), 1)
        w[y == j] = S / (2.0 * n_j)
    return w


def sample_weights(probs: np.ndarray, use_confidence=True, use_class_balance=True) -> np.ndarray:
    y = binary_labels(probs)
    w = np.ones(len(probs), dtype=np.float32)
    if use_confidence:
        w *= confidence_weights(probs)
    if use_class_balance:
        w *= class_balance_weights(y)
    return w


def stratified_split(probs, types, test_size=0.25, seed=42):
    """75/25 split, stratified on (defect-prob bin, wafer type) as in the paper.

    Returns boolean masks (train_mask, test_mask).
    """
    from sklearn.model_selection import train_test_split
    strata = np.array([f"{p:.2f}_{t}" for p, t in zip(probs, types)])
    idx = np.arange(len(probs))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strata)
    train_mask = np.zeros(len(probs), dtype=bool)
    test_mask = np.zeros(len(probs), dtype=bool)
    train_mask[tr] = True
    test_mask[te] = True
    return train_mask, test_mask


if __name__ == "__main__":
    # Self-check: label/weight logic without needing the image files.
    probs = np.array([0.0, 0.3333333, 0.6666667, 1.0], dtype=np.float32)
    assert list(binary_labels(probs)) == [0, 0, 1, 1], binary_labels(probs)
    cw = confidence_weights(probs)
    assert np.allclose(cw, [1.0, 0.33, 0.67, 1.0]), cw
    y = binary_labels(probs)
    cb = class_balance_weights(y)               # 2 per class, S=4 -> all 1.0
    assert np.allclose(cb, 1.0), cb
    # Imbalanced: 3 functional, 1 defective -> functional .667, defective 2.0
    cb2 = class_balance_weights(np.array([0, 0, 0, 1]))
    assert np.allclose(cb2, [4/6, 4/6, 4/6, 4/2]), cb2
    print("data.py self-check OK")
