# Reproducing GBH-YOLOv5 on PV-Multi-Defect

Paper: Li, Wang, Zhang. *GBH-YOLOv5* — Electronics 2023, 12, 561. DOI 10.3390/electronics12030561.

## What we reproduce (Table 4, ablation)

| Run cfg | Model | Paper mAP | P | R |
|---|---|---|---|---|
| `pv_yolov5s.yaml`     | YOLOv5s baseline            | 78.1 | 83.2 | 73.4 |
| `pv_yolov5s_1.yaml`   | + BottleneckCSP             | 94.2 | 88.2 | 90.5 |
| `pv_yolov5s_2.yaml`   | + extra tiny-target head    | 97.1 | 93.4 | 94.6 |
| `pv_gbh_yolov5s.yaml` | + GhostConv = **GBH-YOLOv5**| 97.8 | 96.4 | 93.3 |

Settings from the paper: Adam, lr 0.001, batch 16, 500 epochs, COCO-pretrained
`yolov5s.pt`, 5 classes, 600×600 images, 886 train / 222 val.

## 1. Environment (uv)

```bash
cd GBH-YOLOv5
uv venv --python 3.10
# CPU (smoke test / model build only):
uv pip install -e .
# GPU — install the torch build matching your CUDA, e.g. CUDA 11.6:
uv pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu116
uv pip install -e .
```

Note: this is YOLOv5 **v5.0-era** code (Focus layer, `hyp.scratch`). It needs
`numpy<1.24` and a torch that still exposes the v5.0 APIs — hence torch 1.12.x.
If your GPU is newer (Ada/Hopper/Blackwell) and needs CUDA ≥ 11.8 / torch ≥ 2.0,
that torch drops some APIs this code uses; you'll either patch a few call sites
in `utils/` or port the four `models/pv_*.yaml` onto current Ultralytics YOLOv5.
The yaml configs and data prep here are framework-version independent.

## 2. Data

The dataset is not committed. Download it from the portal, then convert:

```bash
uv run python download_dataset.py      # -> ./PV-Multi-Defect (from GitHub portal)
uv run python prepare_pv_data.py       # -> datasets/pv/{images,labels}/{train,val}
```
`data/pv.yaml` points at `datasets/pv`. Portal:
https://github.com/CCNUZFW/PV-Multi-Defect

## 3. Pretrained weights

Download `yolov5s.pt` (v5.0 release) into the repo root, or pass `--weights ''`
to train from scratch (paper uses COCO-pretrained).

## 4. Train the ablation

```bash
bash run_ablation.sh                 # all four, 500 epochs
EPOCHS=150 bash run_ablation.sh      # faster pass on a smaller GPU
```
On <8 GB VRAM drop the batch: `BATCH=4 bash run_ablation.sh`.
On low system RAM (<16 GB) also set `WORKERS=0` — Windows spawns a fresh torch
import per dataloader worker, which OOMs small machines. Force CPU with `DEVICE=cpu`.

## 5. Metrics

Each run writes `runs/train/<name>/results.txt` (per-epoch P/R/mAP@0.5) and
`weights/best.pt`. Per-class mAP (paper Table 5):

```bash
uv run python test.py --data data/pv.yaml --img 640 \
  --weights runs/train/pv_gbh/weights/best.pt --verbose
```

## Deviations from the paper (honest notes)
- Image size 640 (nearest stride-32 multiple of the 600 px source), not 600.
- Split is a deterministic seed-0 80/20 (~886/221), not the paper's exact list.
- Neck kept as C3 across all four variants; the three ablated axes are backbone
  block (C3→BottleneckCSP), the extra P2 head, and downsample (Conv→GhostConv).
  This matches the authors' released `models/yolov5s_pv.yaml` final model.
