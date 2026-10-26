#!/usr/bin/env python3
"""Train LPSAD.

    python train.py -c config/lpsad-darknet53.yaml -d /data/semantic-kitti \
        -l logs/lpsad [-p /models/darknet53] [--resume logs/lpsad/last.pt]

@ingroup cli
"""

import argparse
import shutil
from pathlib import Path

from lpsad.config import build_label_space, load_config
from lpsad.trainer import Trainer


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("@ingroup")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config/lpsad-darknet53.yaml")
    ap.add_argument("-d", "--dataset", help="overrides train.data_root")
    ap.add_argument("-l", "--logdir", required=True)
    ap.add_argument("-p", "--pretrained", help="RangeNet++ model dir (backbone init)")
    ap.add_argument("--resume", help="checkpoint to resume from")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. train.batch_size=2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["train"]["data_root"] = args.dataset
    if args.pretrained:
        cfg["model"]["pretrained"] = args.pretrained
    for item in args.set:
        apply_override(cfg, item)

    labels = build_label_space(cfg)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, logdir / "config.yaml")
    Trainer(cfg, labels, logdir, resume=args.resume).fit()


def apply_override(cfg: dict, item: str):
    """Apply one ``key.sub=value`` override to the config, in place.

    @param cfg   configuration dict
    @param item  dotted key and YAML value, e.g. ``train.batch_size=2``
    """
    import yaml
    key, value = item.split("=", 1)
    *path, last = key.split(".")
    node = cfg
    for p in path:
        node = node[p]
    node[last] = yaml.safe_load(value)


if __name__ == "__main__":
    main()
