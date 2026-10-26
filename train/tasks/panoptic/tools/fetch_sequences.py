#!/usr/bin/env python3
"""Download selected KITTI odometry sequences without the full 85 GB archive.

The KITTI velodyne archive is one zip file. Its server supports HTTP range
requests, so individual scans can be read straight out of it (``remotezip``).
The matching SemanticKITTI labels are extracted for exactly the same scans,
from the labels archive (a URL, or a local copy of data_odometry_labels.zip,
about 180 MB). Both archives start with ``dataset/``, so the dataset root to
pass to the other scripts is ``OUT/dataset``.

    python tools/fetch_sequences.py -o /data/kitti -s 08 --max-scans 500
    python tools/fetch_sequences.py -o /data/kitti -s 03 04 07 --labels ~/data_odometry_labels.zip

Sizes (velodyne only): 00 8.8 GB, 01 1.9, 02 9.4, 03 1.6, 04 0.55, 05 5.5,
06 2.2, 07 2.1, 08 8.0, 09 3.2, 10 2.4 GB.

KITTI data is licensed for non-commercial use; register and accept the terms
at https://www.cvlibs.net/datasets/kitti/eval_odometry.php before downloading.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

VELODYNE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
LABELS_URL = "https://www.semantic-kitti.org/assets/data_odometry_labels.zip"


def select(names: list[str], sequence: str, max_scans: int | None) -> list[str]:
    """Scan files of one sequence, optionally an evenly spaced subset."""
    prefix = f"dataset/sequences/{sequence}/velodyne/"
    scans = sorted(n for n in names if n.startswith(prefix) and n.endswith(".bin"))
    if max_scans is not None and len(scans) > max_scans:
        idx = np.linspace(0, len(scans) - 1, max_scans).round().astype(int)
        scans = [scans[i] for i in idx]
    return scans


def open_archive(location: str):
    """A zip from a URL (read with HTTP range requests) or a local path."""
    if location.startswith(("http://", "https://")):
        try:
            from remotezip import RemoteZip
        except ImportError:
            sys.exit("needs remotezip: pip install remotezip")
        return RemoteZip(location)
    return zipfile.ZipFile(location)


def extract(archive, names: list[str], out: Path, what: str):
    for i, name in enumerate(names):
        target = out / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(archive.read(name))
        tmp.rename(target)
        if (i + 1) % 100 == 0:
            print(f"  {what} {i + 1}/{len(names)}", flush=True)


def label_name(scan: str) -> str:
    return scan.replace("/velodyne/", "/labels/").removesuffix(".bin") + ".label"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True, help="dataset root (gets sequences/)")
    ap.add_argument("-s", "--sequences", nargs="+", required=True, help="e.g. 08 or 03 04")
    ap.add_argument("--max-scans", type=int, help="evenly spaced subset per sequence")
    ap.add_argument("--velodyne", default=VELODYNE_URL, help="URL or local zip")
    ap.add_argument("--labels", default=LABELS_URL,
                    help="URL or local zip; 'none' to skip (test sequences have no labels)")
    args = ap.parse_args()

    out = Path(args.out)
    with open_archive(args.velodyne) as archive:
        names = archive.namelist()
        selected = {}
        for seq in (f"{int(s):02d}" for s in args.sequences):
            selected[seq] = select(names, seq, args.max_scans)
            print(f"sequence {seq}: {len(selected[seq])} scans")
            extract(archive, selected[seq], out, "scans")

    if args.labels != "none":
        with open_archive(args.labels) as archive:
            available = set(archive.namelist())
            for seq, scans in selected.items():
                wanted = [label_name(s) for s in scans if label_name(s) in available]
                print(f"sequence {seq}: {len(wanted)} label files")
                extract(archive, wanted, out, "labels")
    print(f"dataset root: {out / 'dataset'}")


if __name__ == "__main__":
    main()
