"""Build each ablation model on CPU and run one forward pass. Fails loud if a
yaml or module is wrong — cheap check before spending GPU on training."""
import torch
from models.yolo import Model

CFGS = [
    ("pv_yolov5s",     "models/pv_yolov5s.yaml",     3),
    ("pv_yolov5s_1",   "models/pv_yolov5s_1.yaml",   3),
    ("pv_yolov5s_2",   "models/pv_yolov5s_2.yaml",   4),
    ("pv_gbh_yolov5s", "models/pv_gbh_yolov5s.yaml", 4),
]

x = torch.zeros(1, 3, 640, 640)
for name, cfg, n_heads in CFGS:
    m = Model(cfg).eval()
    assert len(m.model[-1].anchors) == n_heads, f"{name}: head count mismatch"
    with torch.no_grad():
        _, feats = m(x)  # (inference_out, list_of_head_feature_maps)
    assert len(feats) == n_heads, f"{name}: got {len(feats)} heads, want {n_heads}"
    nc = feats[0].shape[-1] - 5
    assert nc == 5, f"{name}: nc={nc}, want 5"
    print(f"OK  {name:16s} heads={n_heads} nc={nc}")
print("all models build + forward-pass clean")
