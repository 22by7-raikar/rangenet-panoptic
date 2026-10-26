#!/usr/bin/env python3
"""Training-free panoptic baseline: semantic predictions + Euclidean clustering.

Takes any semantic segmentation in SemanticKITTI format, for example the
predictions RangeNet++ writes with ``train/tasks/semantic/infer.py`` or the
ones its authors published, splits every thing class into instances with
lpsad.geometric.euclidean_clusters(), writes panoptic labels, and evaluates
them. No training and no GPU needed.

    # RangeNet++ semantics + clustering
    python tools/baseline.py -d /data/semantic-kitti -p rangenet_preds -o preds_geo -s 08
    # ground-truth semantics: how good is the clustering alone?
    python tools/baseline.py -d /data/semantic-kitti --semantics gt -o preds_oracle -s 08

Use ``--radius`` (metres, all thing classes) and ``--class-radius
person=0.3`` to tune the linking distance.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpsad.config import build_label_space, load_config  # noqa: E402
from lpsad.geometric import euclidean_clusters  # noqa: E402
from lpsad.metrics import PanopticEval  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "lpsad-darknet53.yaml"


def summary(res: dict) -> dict:
    keys = ("PQ", "PQ_dagger", "RQ", "SQ", "PQ_th", "RQ_th", "SQ_th", "PQ_st", "mIoU")
    return {k: round(100 * res[k], 2) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True,
                    help="root with sequences/XX/{velodyne,labels}")
    ap.add_argument("-p", "--predictions",
                    help="semantic predictions root (sequences/XX/predictions)")
    ap.add_argument("--semantics", choices=["pred", "gt"], default="pred")
    ap.add_argument("-o", "--out", help="write panoptic .label files here")
    ap.add_argument("-s", "--sequences", nargs="+", default=["08"])
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--class-radius", nargs="*", default=[], metavar="NAME=METRES")
    ap.add_argument("--min-points", type=int, default=1)
    ap.add_argument("--max-scans", type=int, help="evenly spaced subset per sequence")
    ap.add_argument("--json", help="also write the metrics here")
    args = ap.parse_args()
    if args.semantics == "pred" and not args.predictions:
        ap.error("--predictions is required with --semantics pred")

    labels = build_label_space(load_config(args.config))
    radius = {c: args.radius for c in labels.things}
    for item in args.class_radius:
        name, value = item.split("=")
        radius[labels.names.index(name)] = float(value)

    ev = PanopticEval(labels.nclasses, labels.ignore, labels.things)
    count = 0
    for seq in (f"{int(s):02d}" for s in args.sequences):
        root = Path(args.dataset) / "sequences" / seq
        scans = sorted((root / "velodyne").glob("*.bin"))
        if args.max_scans and len(scans) > args.max_scans:
            idx = np.linspace(0, len(scans) - 1, args.max_scans).round().astype(int)
            scans = [scans[i] for i in idx]
        for scan in scans:
            xyz = np.fromfile(scan, dtype=np.float32).reshape(-1, 4)[:, :3]
            gt_file = root / "labels" / f"{scan.stem}.label"
            gt = np.fromfile(gt_file, dtype=np.uint32).astype(np.int64)
            if args.semantics == "gt":
                raw = gt
            else:
                pred_dir = Path(args.predictions) / "sequences" / seq / "predictions"
                pred_file = pred_dir / f"{scan.stem}.label"
                raw = np.fromfile(pred_file, dtype=np.uint32).astype(np.int64)
            sem = labels.learning_lut[raw & 0xFFFF]
            inst = euclidean_clusters(xyz, sem, labels.things, radius, args.min_points)
            ev.add(sem, inst, labels.learning_lut[gt & 0xFFFF], gt >> 16)
            if args.out:
                path = Path(args.out) / "sequences" / seq / "predictions" / f"{scan.stem}.label"
                path.parent.mkdir(parents=True, exist_ok=True)
                (labels.inverse_lut[sem] | (inst << 16)).astype(np.uint32).tofile(path)
            count += 1

    res = summary(ev.results())
    print(f"evaluated {count} scans ({args.semantics} semantics, radius {args.radius} m)")
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"args": vars(args), "metrics": res}, indent=2))


if __name__ == "__main__":
    main()
