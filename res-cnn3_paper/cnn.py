"""Deep regression CNN (modified VGG-19) from Deitsch et al. 2019, Sec. 3.2.

VGG-19 conv base (ImageNet) -> Global Average Pooling -> FC 4096 -> FC 2048 ->
1 sigmoid output = defect probability. Trained by minimizing MSE against the
continuous label p in {0, .33, .67, 1}. Two-stage schedule: (1) Adam on the new
head with the conv base frozen, (2) SGD refine of all layers. Predictions are
thresholded at 0.5 for the binary functional/defective decision.

GPU wiring: uses CUDA when available, with automatic mixed precision and
gradient accumulation so the ~300x300 inputs fit a small (4 GB) card while
keeping the paper's effective mini-batch of 16.

Run:  python cnn.py --root elpv-dataset --epochs-head 20 --epochs-all 80
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights

import data as D

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ELPVSet(Dataset):
    def __init__(self, imgs, probs, weights, size=300, train=False):
        self.imgs = imgs            # (N, H, W) uint8 grayscale
        self.probs = probs.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.train = train
        # Moderate augmentation (Sec. 3.2): +/-3 deg rotation, +/-2% shift/scale,
        # random flips, and random 90 deg rotations (busbars are symmetric).
        aug = []
        if train:
            aug += [
                T.RandomChoice([T.RandomRotation((0, 0)), T.RandomRotation((90, 90)),
                                T.RandomRotation((180, 180)), T.RandomRotation((270, 270))]),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            ]
        self.tf = T.Compose(aug + [
            T.Grayscale(num_output_channels=3),             # gray -> 3ch for VGG (picklable)
            T.ToTensor(),                                   # 3xHxW in [0,1]
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.size = size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.fromarray(self.imgs[i])                 # LxL grayscale
        x = self.tf(img)
        return x, self.probs[i], self.weights[i]


class VGG19Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = base.features                       # conv base
        self.gap = nn.AdaptiveAvgPool2d(1)                  # Global Average Pooling
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 1), nn.Sigmoid(),               # defect probability
        )

    def forward(self, x):
        return self.head(self.gap(self.features(x))).squeeze(1)

    def set_base_trainable(self, flag: bool):
        for p in self.features.parameters():
            p.requires_grad_(flag)


def weighted_mse(pred, target, w):
    return (w * (pred - target) ** 2).mean()


def run_epoch(model, loader, device, optimizer=None, scaler=None, accum=1):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, (x, y, w) in enumerate(loader):
        x, y, w = x.to(device), y.to(device), w.to(device)
        with torch.set_grad_enabled(train), torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(x)
            loss = weighted_mse(pred, y, w)
        if train:
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(pred, probs, types, tag=""):
    from sklearn.metrics import roc_auc_score, accuracy_score
    y = D.binary_labels(probs)
    yhat = (pred > 0.5).astype(int)
    lines = []
    for name, mask in [("overall", np.ones(len(y), bool)),
                       ("mono", types == "mono"), ("poly", types == "poly")]:
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y[mask], yhat[mask])
        try:
            auc = roc_auc_score(y[mask], pred[mask])
        except ValueError:
            auc = float("nan")
        lines.append(f"  {tag}{name:8s}  acc={acc*100:5.2f}%  auc={auc*100:5.2f}%  n={mask.sum()}")
    print("\n".join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="elpv-dataset/src/elpv_dataset/data")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="fits ~4GB at 300px; accum reaches 16")
    ap.add_argument("--accum", type=int, default=2, help="grad accumulation steps")
    ap.add_argument("--epochs-head", type=int, default=20)
    ap.add_argument("--epochs-all", type=int, default=80)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-all", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="cnn_vgg19.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (CPU: training will be slow)"))

    paths, probs, types = D.load_labels(args.root)
    imgs = D.load_images(args.root, paths, size=args.size)
    tr, te = D.stratified_split(probs, types, seed=args.seed)
    w = D.sample_weights(probs)                             # confidence * class balance

    tr_set = ELPVSet(imgs[tr], probs[tr], w[tr], args.size, train=True)
    te_set = ELPVSet(imgs[te], probs[te], np.ones(te.sum(), np.float32), args.size, train=False)
    pin = device.type == "cuda"
    tr_loader = DataLoader(tr_set, args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin, drop_last=True)
    te_loader = DataLoader(te_set, args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = VGG19Regressor().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Stage 1: frozen conv base, Adam on the head (Sec. 3.2).
    model.set_base_trainable(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr_head, betas=(0.9, 0.999), weight_decay=1e-8)
    for e in range(args.epochs_head):
        loss = run_epoch(model, tr_loader, device, opt, scaler, args.accum)
        print(f"[head {e+1}/{args.epochs_head}] mse={loss:.4f}")

    # Stage 2: refine all layers with SGD.
    model.set_base_trainable(True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_all, momentum=0.9)
    for e in range(args.epochs_all):
        loss = run_epoch(model, tr_loader, device, opt, scaler, args.accum)
        if (e + 1) % 5 == 0 or e == 0:
            print(f"[all {e+1}/{args.epochs_all}] mse={loss:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}")
    pred = predict(model, te_loader, device)
    print("CNN test results:")
    evaluate(pred, probs[te], types[te])


if __name__ == "__main__":
    main()
