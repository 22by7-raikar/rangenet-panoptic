#!/usr/bin/env python3
"""Write SemanticKITTI panoptic predictions with a trained LPSAD checkpoint.

    python infer.py -m logs/lpsad/best.pt -d /data/semantic-kitti -o preds \
        --split valid

Clustering settings can be changed per run (paper Table IV ablations):

    python infer.py ... --cluster features=center learned_radius=false

Labels go to <out>/sequences/XX/predictions/NNNNNN.label as
``semantic_id | (instance_id << 16)``, the format of the official benchmark
and semantic-kitti-api.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from lpsad.cluster import panoptic_inference
from lpsad.config import build_label_space
from lpsad.dataset import SemanticKittiPanoptic, collate
from lpsad.model import LPSAD
from lpsad.trainer import autocast_context, to_device


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="checkpoint (.pt)")
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cluster", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override clustering settings without retraining, e.g. "
                         "features=center learned_radius=false fixed_center_radius=1.0")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    for item in args.cluster:
        key, value = item.split("=", 1)
        cfg["cluster"][key] = yaml.safe_load(value)
    labels = build_label_space(cfg)
    model = LPSAD(cfg, labels.nclasses, labels.ncategories).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scale = tuple(ckpt["radius_scale"])
    if cfg["cluster"]["radius_scale"] != "auto":
        scale = (float(cfg["cluster"]["radius_scale"]),) * 2
    is_thing = torch.from_numpy(labels.is_thing()).to(device)
    inverse = torch.from_numpy(labels.inverse_lut).to(device)
    print(f"epoch {ckpt['epoch']}, radius scale {scale}")

    ds = SemanticKittiPanoptic(args.dataset, cfg["train"]["split"][args.split],
                               cfg, labels, gt=False)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate)
    out_root = Path(args.out) / "sequences"
    gen = torch.Generator().manual_seed(0)
    times = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            with autocast_context(cfg["train"]["amp"], device):
                out = model(batch["input"], batch["xyz"], batch["mask"])
            out = {k: v.float() for k, v in out.items()}
            preds = panoptic_inference(out, batch, is_thing, cfg["cluster"], scale, gen)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - start) / len(preds))
            for (sem, inst), seq, name in zip(preds, batch["seq"], batch["name"]):
                label = inverse[sem] | (inst << 16)
                path = out_root / seq / "predictions" / f"{name}.label"
                path.parent.mkdir(parents=True, exist_ok=True)
                label.cpu().numpy().astype(np.uint32).tofile(path)
    print(f"{len(ds)} scans, {1000 * np.mean(times):.1f} ms/scan "
          f"(network + clustering)")


if __name__ == "__main__":
    main()
