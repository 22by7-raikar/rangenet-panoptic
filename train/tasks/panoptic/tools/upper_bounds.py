#!/usr/bin/env python3
"""How much does the range image itself limit panoptic quality? (no training)

Uses ground truth only, at several image widths:

1. **Projection.** Share of points that own a pixel, and thing instances
   that keep fewer than ``--visible`` points. Then PQ when every pixel has its
   true class and instance and every point takes its pixel's labels (paper
   Sec. III-E): the ceiling for a perfect network without kNN.
2. **Clustering.** True classes and true centre offsets plus Gaussian noise
   (``--noise`` metres per axis) go through the LPSAD instance extractor. The
   learned radius is replaced by its ideal value, each pixel's actual offset
   error (at least ``--radius-floor``). Compares radius modes ``seed``
   (paper), ``sum`` and a fixed radius.

    python tools/upper_bounds.py -d /data/semantic-kitti -s 08 --max-scans 200 \
        --widths 512 1024 2048 --noise 0 0.1 0.2 0.4 --json bounds.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpsad.cluster import extract_instances  # noqa: E402
from lpsad.config import build_label_space, load_config  # noqa: E402
from lpsad.dataset import SemanticKittiPanoptic  # noqa: E402
from lpsad.metrics import PanopticEval  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "lpsad-darknet53.yaml"


def pct(res, keys=("PQ", "RQ", "SQ", "PQ_th", "RQ_th", "SQ_th", "PQ_st", "mIoU")):
    return {k: round(100 * res[k], 2) for k in keys}


def cluster_modes(fixed_radius: float):
    return {
        "seed": dict(learned_radius=True, radius_mode="seed"),
        "sum": dict(learned_radius=True, radius_mode="sum"),
        f"fixed {fixed_radius} m": dict(learned_radius=False, center_radius=fixed_radius),
    }


def run_width(cfg, labels, args, width, gen):
    cfg = copy.deepcopy(cfg)
    cfg["sensor"]["width"] = width
    ds = SemanticKittiPanoptic(args.dataset, args.sequences, cfg, labels, gt=True,
                               max_scans=args.max_scans)
    is_thing = torch.from_numpy(labels.is_thing())
    modes = cluster_modes(args.fixed_radius)
    proj_eval = PanopticEval(labels.nclasses, labels.ignore, labels.things)
    evals = {(s, m): PanopticEval(labels.nclasses, labels.ignore, labels.things)
             for s in args.noise for m in modes}
    visible, total_pts, inst_total, inst_lost = 0, 0, 0, 0

    for item in ds:
        px, py = item["px"], item["py"]
        gt_sem, gt_inst = item["point_sem"].numpy(), item["point_inst"].numpy()
        mask = item["mask"]

        # 1. projection ceiling
        visible += int(mask.sum())
        total_pts += len(gt_sem)
        owner_inst = item["inst"][mask]
        ids, counts = np.unique(gt_inst[gt_inst > 0], return_counts=True)
        big = ids[counts >= args.min_points]
        seen = dict(zip(*np.unique(owner_inst[owner_inst > 0].numpy(), return_counts=True),
                        strict=True))
        inst_total += len(big)
        inst_lost += sum(seen.get(i, 0) < args.visible for i in big)
        proj_eval.add(item["sem"][py, px].numpy(), item["inst"][py, px].numpy(), gt_sem, gt_inst)

        # 2. clustering with noisy oracle offsets
        sem = item["sem"]
        for sigma in args.noise:
            noise = torch.randn(item["offset"].shape, generator=gen) * sigma
            offset = item["offset"] + noise
            # ideal learned radius: each pixel's actual error (floored: a trained
            # network never predicts exactly 0, and 0 splits objects on rounding)
            err = noise.norm(dim=0, keepdim=True).clamp_min(args.radius_floor)
            for name, kw in modes.items():
                inst = extract_instances(sem, mask, is_thing, item["xyz"], offset, err,
                                         torch.zeros(1, *sem.shape), err, features="center",
                                         generator=gen, **kw)
                evals[(sigma, name)].add(sem[py, px].numpy(), inst[py, px].numpy(),
                                         gt_sem, gt_inst)

    return {
        "width": width,
        "scans": len(ds),
        "points_visible_pct": round(100 * visible / total_pts, 2),
        "instances": inst_total,
        f"instances_below_{args.visible}_visible_points_pct":
            round(100 * inst_lost / max(inst_total, 1), 2),
        "projection_ceiling": pct(proj_eval.results()),
        "clustering": {f"noise {s} m / {m}": pct(e.results(), ("PQ_th", "RQ_th", "SQ_th"))
                       for (s, m), e in evals.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-s", "--sequences", nargs="+", type=int, default=[8])
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--widths", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.4])
    ap.add_argument("--fixed-radius", type=float, default=0.5)
    ap.add_argument("--radius-floor", type=float, default=0.05,
                    help="lower bound of the ideal learned radius, metres")
    ap.add_argument("--max-scans", type=int, default=200)
    ap.add_argument("--min-points", type=int, default=50, help="instances counted (as evaluation)")
    ap.add_argument("--visible", type=int, default=10, help="'lost' below this many pixels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = build_label_space(cfg)
    gen = torch.Generator().manual_seed(args.seed)
    results = [run_width(cfg, labels, args, w, gen) for w in args.widths]
    print(json.dumps(results, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"args": vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
