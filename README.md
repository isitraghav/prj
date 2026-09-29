# PV defect papers — reproductions

Three PV-defect papers, each in its own folder with the paper PDF and code.
Datasets are **not** in git (too big); fetch them with `setup_datasets.sh`.

| Folder | Paper | Code status |
|--------|-------|-------------|
| `res-cnn3_paper` | Deitsch et al. 2019, EL cell classification (VGG-19 + SVM) | built, GPU-ready |
| `elseg_paper` | Fioresi et al. 2022, EL semantic segmentation (DeepLabv3) | upstream UCF code |
| `udensenet_paper` | Pamungkas et al. 2023, IR fault classification (Coupled UDenseNet) | dataset only, code TODO |

## Run on a fresh machine
```bash
git clone https://github.com/isitraghav/prj.git && cd prj
bash setup_datasets.sh                       # re-fetch datasets (git + one unzip)

# GPU PyTorch (CPU wheel is default; install the CUDA build to use a GPU):
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r res-cnn3_paper/requirements.txt

# res-cnn3 reproduction (fast GPU config):
cd res-cnn3_paper
python cnn.py --size 224 --batch 8 --accum 2 --epochs-head 6 --epochs-all 10   # CNN
python svm.py                                                                   # SVM
```
`cnn.py` auto-uses CUDA when available (AMP + gradient accumulation to fit a
4 GB card). See `res-cnn3_paper/README.md` for full details and paper targets.
