# Deitsch et al. 2019 — ELPV defect classification (reproduction)

Reproduces the two classifiers from *"Automatic classification of defective
photovoltaic module cells in electroluminescence images"* (Solar Energy 185:455–468):

| Approach | File | Method | Paper (overall) |
|----------|------|--------|-----------------|
| Deep regression CNN | `cnn.py` | VGG-19 + GAP + FC4096 + FC2048 + sigmoid, MSE regression on defect probability | acc 88.42 %, AUC 94.42 % |
| Hand-crafted SVM | `svm.py` | KAZE/VGG local features → VLAD → linear SVM | acc ≈82 %, AUC 88.51 % |

`data.py` handles the ELPV dataset: labels `p ∈ {0, .33, .67, 1}`, binary target
`defective = p > 0.5`, the 75/25 stratified split (→ 1968 train / 656 test,
matching Table 3), Table 2 confidence weights, and the `S/2nⱼ` class balance (Eq. 5).

## Dataset
Cloned to `elpv-dataset/` (`git clone https://github.com/zae-bayern/elpv-dataset`).
Data root: `elpv-dataset/src/elpv_dataset/data` (the scripts default to it).

## Setup
```bash
pip install -r requirements.txt          # numpy, pillow, scikit-learn, opencv-contrib-python
# torchvision is required by cnn.py and is NOT in the default torch wheel.
# This machine has an RTX 2050 (4 GB, driver 591.86, CUDA 13) but a CPU-only
# torch. Install the CUDA build so training uses the GPU:
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## Run
```bash
python cnn.py     # full paper schedule: 20 head epochs (Adam) + 80 all (SGD), 100 total
python svm.py     # KAZE/VGG → VLAD → grid-searched linear SVM
python data.py    # label/weight self-check (no dataset images needed)
```

Both scripts print `overall / mono / poly` accuracy and ROC AUC on the test set.

## GPU wiring (cnn.py)
- Auto-selects CUDA when available, else CPU (with a warning).
- **Mixed precision** (`torch.autocast` + `GradScaler`) + **gradient accumulation**
  (`--batch 8 --accum 2` → effective 16, the paper's mini-batch) to fit the 4 GB card.
  If you still hit OOM: lower `--batch` to 4 (`--accum 4`), or `--size 224`.

## Known deviations from the paper
- **SVM features**: the paper's best is KAZE keypoints + VGG descriptor. The OpenCV
  build here (5.0, slim) ships neither KAZE nor `xfeatures2d`, so `svm.py` falls back
  to **SIFT/SIFT** — also evaluated in the paper (Fig. 6b/7, ~87 % AUC). Install a full
  `opencv-contrib` build with KAZE + `xfeatures2d.VGG` to get the exact variant; the code
  auto-detects and uses it.
- **VLAD**: single codebook, not the 5-codebook + PCA-whitening ensemble (Sec. 3.1.4) —
  marginal gain, see the `ponytail:` note in `svm.py`.
