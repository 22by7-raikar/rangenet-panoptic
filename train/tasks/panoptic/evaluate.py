#!/usr/bin/env python3
"""Evaluate panoptic predictions (PQ, PQ-dagger, RQ, SQ, mIoU) on points.

    python evaluate.py -d /data/semantic-kitti -p preds --split valid

For numbers you report, cross-check with the official semantic-kitti-api
``evaluate_panoptic.py``.

@ingroup cli
"""

import argparse
from pathlib import Path

import numpy as np

from lpsad.config import build_label_space, load_config
from lpsad.metrics import PanopticEval


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("@ingroup")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-p", "--predictions", required=True)
    ap.add_argument("-c", "--config", default="config/lpsad-darknet53.yaml")
    ap.add_argument("--split", default="valid", choices=["train", "valid"])
    ap.add_argument("--min-points", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = build_label_space(cfg)
    ev = PanopticEval(labels.nclasses, labels.ignore, labels.things, args.min_points)
    count = 0
    for seq in cfg["train"]["split"][args.split]:
        seq = f"{int(seq):02d}"
        root = Path(args.dataset) / "sequences" / seq
        pred_dir = Path(args.predictions) / "sequences" / seq / "predictions"
        # the scans present define the evaluation set (subsets are allowed)
        for scan in sorted((root / "velodyne").glob("*.bin")):
            gt_file = root / "labels" / f"{scan.stem}.label"
            pred_file = pred_dir / gt_file.name
            if not pred_file.exists():
                raise FileNotFoundError(pred_file)
            gt = np.fromfile(gt_file, dtype=np.uint32).astype(np.int64)
            pred = np.fromfile(pred_file, dtype=np.uint32).astype(np.int64)
            ev.add(labels.learning_lut[pred & 0xFFFF], pred >> 16,
                   labels.learning_lut[gt & 0xFFFF], gt >> 16)
            count += 1

    res = ev.results()
    print(f"evaluated {count} scans")
    for k in ("PQ", "PQ_dagger", "RQ", "SQ", "PQ_th", "RQ_th", "SQ_th",
              "PQ_st", "RQ_st", "SQ_st", "mIoU"):
        print(f"{k:>10}: {100 * res[k]:6.2f}")
    print(f"\n{'class':>15} {'PQ':>6} {'RQ':>6} {'SQ':>6} {'IoU':>6}")
    pc = res["per_class"]
    for c in ev.include:
        print(f"{labels.names[c]:>15} " + " ".join(
            f"{100 * pc[m][c]:6.2f}" for m in ("PQ", "RQ", "SQ", "IoU")))


if __name__ == "__main__":
    main()
