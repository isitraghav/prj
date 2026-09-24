#!/usr/bin/env bash
# Reproduce Table 4 (ablation) of GBH-YOLOv5 on PV-Multi-Defect.
# Paper settings: Adam, lr 0.001, batch 16, 500 epochs, COCO-pretrained yolov5s.pt.
# Run: bash run_ablation.sh   (from repo root, after prepare_pv_data.py)
set -e

EPOCHS=${EPOCHS:-500}
BATCH=${BATCH:-16}
IMG=${IMG:-640}          # dataset is 600x600; 640 = nearest stride-32 multiple
WEIGHTS=${WEIGHTS:-yolov5s.pt}
DEVICE=${DEVICE:-0}      # GPU id; set DEVICE=cpu to force CPU
WORKERS=${WORKERS:-8}   # dataloader workers; set WORKERS=0 on low-RAM machines
DATA=data/pv.yaml
HYP=data/hyp.pv.yaml

train () {  # $1=cfg  $2=run_name
  echo "=== training $2 ==="
  uv run python train.py \
    --data "$DATA" --cfg "$1" --hyp "$HYP" --weights "$WEIGHTS" \
    --img "$IMG" --batch-size "$BATCH" --epochs "$EPOCHS" \
    --device "$DEVICE" --workers "$WORKERS" \
    --adam --name "$2"
}

train models/pv_yolov5s.yaml     pv_yolov5s      # row 1: baseline
train models/pv_yolov5s_1.yaml   pv_yolov5s_1    # row 2: +BottleneckCSP
train models/pv_yolov5s_2.yaml   pv_yolov5s_2    # row 3: +extra head
train models/pv_gbh_yolov5s.yaml pv_gbh          # row 4: +GhostConv (final)

echo "done. per-run metrics: runs/train/<name>/results.txt and best.pt"
