"""Geometric clustering and the training-free tools (baseline, bounds, benchmark)."""

import json
import subprocess
import sys

import numpy as np
from conftest import PANOPTIC

from lpsad.geometric import euclidean_clusters

sys.path.insert(0, str(PANOPTIC / "tools"))
from fetch_sequences import select  # noqa: E402


def test_euclidean_clusters_split_by_distance_and_class():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 0.1, (50, 3))
    b = rng.normal([5, 0, 0], 0.1, (50, 3))       # far: separate car
    c = rng.normal([0.05, 0, 0], 0.1, (50, 3))    # overlaps a, but a person
    road = rng.normal([0, 0, -2], 1.0, (50, 3))
    xyz = np.concatenate([a, b, c, road])
    sem = np.array([1] * 100 + [6] * 50 + [9] * 50)
    inst = euclidean_clusters(xyz, sem, things=[1, 6], radius=0.5)
    assert len(set(inst[:50])) == 1 and len(set(inst[50:100])) == 1
    assert len({inst[0], inst[50], inst[100]}) == 3 and (inst[:150] > 0).all()
    assert (inst[150:] == 0).all()
    # per-class radius and min_points
    inst = euclidean_clusters(xyz, sem, things=[1], radius={1: 10.0}, min_points=200)
    assert (inst == 0).all()


def test_fetch_selects_evenly_spaced_scans_of_one_sequence():
    names = ([f"dataset/sequences/08/velodyne/{i:06d}.bin" for i in range(100)]
             + ["dataset/sequences/08/velodyne/", "dataset/sequences/09/velodyne/000000.bin"])
    assert len(select(names, "08", None)) == 100
    picked = select(names, "08", 5)
    assert picked[0].endswith("000000.bin") and picked[-1].endswith("000099.bin")
    assert len(picked) == 5


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=PANOPTIC,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_baseline_with_true_semantics(cfg, tmp_path):
    root = cfg["train"]["data_root"]
    out = run("tools/baseline.py", "-d", root, "--semantics", "gt", "-s", "08",
              "-o", tmp_path / "preds", "--json", tmp_path / "m.json")
    metrics = json.loads((tmp_path / "m.json").read_text())["metrics"]
    # synthetic scans contain 2 of 8 thing classes (max PQ_th 25); objects are
    # placed at random, so the odd touching pair merges, as it would for real
    assert 20.0 < metrics["PQ_th"] <= 25.0 and "evaluated 4 scans" in out
    assert len(list((tmp_path / "preds/sequences/08/predictions").glob("*.label"))) == 4


def test_upper_bounds(cfg, tmp_path):
    run("tools/upper_bounds.py", "-d", cfg["train"]["data_root"], "--widths", "256",
        "--noise", "0", "0.3", "--max-scans", "2", "--json", tmp_path / "b.json")
    (res,) = json.loads((tmp_path / "b.json").read_text())["results"]
    assert 0 < res["points_visible_pct"] < 100
    clus = res["clustering"]
    # with perfect offsets clustering reaches the projection ceiling
    assert clus["noise 0.0 m / seed"]["PQ_th"] == res["projection_ceiling"]["PQ_th"]
    # with noisy offsets the paper's seed radius fragments, seed + candidate doesn't
    assert clus["noise 0.3 m / sum"]["PQ_th"] > clus["noise 0.3 m / seed"]["PQ_th"]


def test_benchmark(cfg, tmp_path):
    run("tools/benchmark.py", "-d", cfg["train"]["data_root"], "--widths", "256",
        "--layers", "21", "--iters", "1", "--warmup", "0", "--json", tmp_path / "t.json")
    (res,) = json.loads((tmp_path / "t.json").read_text())["results"]
    assert res["network_ms"] > 0 and res["clustering_ms"] > 0


def test_fetch_extracts_matching_scans_and_labels(cfg, tmp_path):
    import zipfile
    from pathlib import Path

    root = Path(cfg["train"]["data_root"]) / "sequences"
    velodyne, labels = tmp_path / "velodyne.zip", tmp_path / "labels.zip"
    with zipfile.ZipFile(velodyne, "w") as zv, zipfile.ZipFile(labels, "w") as zl:
        for f in sorted(root.glob("*/velodyne/*.bin")):
            zv.write(f, f"dataset/{f.relative_to(root.parent)}")
        for f in sorted(root.glob("*/labels/*.label")):
            zl.write(f, f"dataset/{f.relative_to(root.parent)}")
    out = tmp_path / "kitti"
    run("tools/fetch_sequences.py", "-o", out, "-s", "08", "--max-scans", "2",
        "--velodyne", velodyne, "--labels", labels)
    seq = out / "dataset" / "sequences" / "08"
    scans = sorted(p.stem for p in (seq / "velodyne").glob("*.bin"))
    assert scans == ["000000", "000003"]
    assert sorted(p.stem for p in (seq / "labels").glob("*.label")) == scans
    assert not (out / "dataset" / "sequences" / "00").exists()
