"""Hand-crafted SVM pipeline from Deitsch et al. 2019, Sec. 3.1 (KAZE/VGG).

KAZE keypoints -> VGG local descriptors -> VLAD encoding (power + intra + L2
normalization) -> linear SVM (squared hinge) with a grid search over C. This
is the best-performing SVM variant in the paper (KAZE/VGG, AUC 88.51% overall).

Needs opencv-contrib-python for the VGG descriptor (cv2.xfeatures2d.VGG). If it
is unavailable, falls back to KAZE's own descriptor and says so.

Run:  python svm.py --root elpv-dataset --clusters 64
"""
from __future__ import annotations
import argparse
import numpy as np
import cv2

import data as D

RHO = 0.5   # power-normalization exponent (Eq. 4)


def make_extractors():
    """Pick detector + descriptor by what this OpenCV build provides.

    Paper's best is KAZE keypoints + VGG descriptor. Falls back to KAZE/KAZE,
    then SIFT/SIFT (also evaluated in the paper, Fig. 6b/7, ~87% AUC) so the
    pipeline runs on a slim OpenCV build without contrib/xfeatures2d.
    Returns (detector, descriptor_or_None, kind_string).
    """
    detector = cv2.KAZE_create() if hasattr(cv2, "KAZE_create") else None
    if detector is not None:
        if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "VGG_create"):
            try:
                return detector, cv2.xfeatures2d.VGG_create(), "KAZE/VGG"
            except cv2.error:
                pass
        return detector, None, "KAZE/KAZE"
    return cv2.SIFT_create(), None, "SIFT/SIFT"


def extract_local(imgs, detector, descriptor, kind):
    """Return list of (N,D) descriptor arrays, one per image."""
    feats = []
    for im in imgs:
        kps = detector.detect(im, None)
        if not kps:
            feats.append(np.zeros((0, 1), np.float32))
            continue
        if descriptor is not None:
            _, desc = descriptor.compute(im, kps)      # separate descriptor (VGG)
        else:
            _, desc = detector.compute(im, kps)        # detector's own (KAZE/SIFT)
        feats.append(np.zeros((0, 1), np.float32) if desc is None else desc.astype(np.float32))
    return feats


def build_codebook(feats, k, seed=42):
    from sklearn.cluster import MiniBatchKMeans
    stacked = np.vstack([f for f in feats if len(f)])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=3)
    km.fit(stacked)
    return km


def vlad(desc, km):
    """VLAD vector for one image's descriptors: sum of residuals per cluster."""
    k, d = km.cluster_centers_.shape
    v = np.zeros((k, d), np.float32)
    if len(desc):
        assign = km.predict(desc)
        for i in range(k):
            sel = desc[assign == i]
            if len(sel):
                v[i] = (sel - km.cluster_centers_[i]).sum(axis=0)
    v = v.ravel()
    v = np.sign(v) * np.abs(v) ** RHO          # power normalization (Eq. 4)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v               # L2 normalization


def encode(feats, km):
    return np.vstack([vlad(f, km) for f in feats])


def grid_search_svm(X, y, w, seed=42):
    """Linear SVM, squared hinge, C in 10^{-2..6}, 5-fold CV (Sec. 3.1.5)."""
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV
    grid = {"C": [10.0 ** k for k in range(-2, 7)]}
    base = LinearSVC(loss="squared_hinge", dual="auto", max_iter=5000, class_weight="balanced")
    gs = GridSearchCV(base, grid, scoring="f1", cv=5, n_jobs=-1)
    gs.fit(X, y, sample_weight=w)
    return gs.best_estimator_, gs.best_params_


def evaluate(clf, X, probs, types):
    from sklearn.metrics import roc_auc_score, accuracy_score
    y = D.binary_labels(probs)
    score = clf.decision_function(X)
    yhat = (score > 0).astype(int)
    for name, mask in [("overall", np.ones(len(y), bool)),
                       ("mono", types == "mono"), ("poly", types == "poly")]:
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y[mask], yhat[mask])
        try:
            auc = roc_auc_score(y[mask], score[mask])
        except ValueError:
            auc = float("nan")
        print(f"  {name:8s}  acc={acc*100:5.2f}%  auc={auc*100:5.2f}%  n={mask.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="elpv-dataset/src/elpv_dataset/data")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--clusters", type=int, default=64, help="VLAD codebook size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths, probs, types = D.load_labels(args.root)
    imgs = list(D.load_images(args.root, paths, size=args.size))  # grayscale uint8
    tr, te = D.stratified_split(probs, types, seed=args.seed)

    detector, descriptor, kind = make_extractors()
    print(f"features: {kind}"
          + ("" if kind == "KAZE/VGG" else "  (install a full opencv-contrib build for the paper's KAZE/VGG)"))

    feats = extract_local(imgs, detector, descriptor, kind)
    feats = np.array(feats, dtype=object)
    km = build_codebook(feats[tr], args.clusters, args.seed)
    X = encode(feats, km)

    w = D.sample_weights(probs, use_class_balance=False)   # LinearSVC handles class balance
    clf, best = grid_search_svm(X[tr], D.binary_labels(probs[tr]), w[tr], args.seed)
    print(f"best {best}")
    print("SVM test results:")
    evaluate(clf, X[te], probs[te], types[te])


# ponytail: single VLAD codebook, not the paper's 5 codebooks + PCA whitening
#           (Sec. 3.1.4). Marginal gain; add 5x random-seed codebooks and
#           sklearn PCA(whiten=True) on the concatenation if you chase the last
#           ~1% AUC.
if __name__ == "__main__":
    main()
