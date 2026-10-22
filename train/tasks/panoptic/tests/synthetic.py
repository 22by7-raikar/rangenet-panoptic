"""Tiny synthetic SemanticKITTI-format scans for tests.

Each scan has a road plane (raw class 40), a building wall (50) and a few
cars (10) and persons (30) with distinct instance ids, all inside the HDL-64
vertical field of view.
"""

from pathlib import Path

import numpy as np


def _box(rng, center, size, n):
    pts = rng.uniform(-0.5, 0.5, size=(n, 3)) * size + center
    return pts


def make_scan(rng, n_cars=3, n_persons=2):
    pts, labels = [], []
    r = rng.uniform(4, 40, 4000)
    a = rng.uniform(-np.pi, np.pi, 4000)
    ground = np.stack([r * np.cos(a), r * np.sin(a), np.full_like(r, -1.73)], 1)
    pts.append(ground)
    labels.append(np.full(len(ground), 40))
    wall = np.stack([np.full(1500, 30.0), rng.uniform(-20, 20, 1500),
                     rng.uniform(-1.7, 2.0, 1500)], 1)
    pts.append(wall)
    labels.append(np.full(len(wall), 50))

    inst = 1
    for cls, count, size, npts in ((10, n_cars, (4.0, 1.8, 1.5), 600),
                                   (30, n_persons, (0.6, 0.6, 1.7), 200)):
        for _ in range(count):
            dist = rng.uniform(6, 20)
            ang = rng.uniform(-np.pi, np.pi)
            c = np.array([dist * np.cos(ang), dist * np.sin(ang), -1.73 + size[2] / 2])
            p = _box(rng, c, np.array(size), npts)
            pts.append(p)
            labels.append(np.full(npts, cls) | (inst << 16))
            inst += 1
    xyz = np.concatenate(pts).astype(np.float32)
    lab = np.concatenate(labels).astype(np.uint32)
    remission = rng.uniform(0, 1, (len(xyz), 1)).astype(np.float32)
    return np.concatenate([xyz, remission], 1), lab


def write_dataset(root: Path, sequences=(0, 8), scans_per_seq=4, seed=0):
    rng = np.random.default_rng(seed)
    for seq in sequences:
        d = Path(root) / "sequences" / f"{seq:02d}"
        (d / "velodyne").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(scans_per_seq):
            scan, lab = make_scan(rng)
            scan.tofile(d / "velodyne" / f"{i:06d}.bin")
            lab.tofile(d / "labels" / f"{i:06d}.label")
    return Path(root)
