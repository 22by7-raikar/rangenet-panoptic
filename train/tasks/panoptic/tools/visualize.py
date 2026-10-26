#!/usr/bin/env python3
"""Figures for slides and docs from real scans.

Three figures, each also writing its numbers as a Markdown table:

``scan``
    One scan as range-image panels (range, remission, and when available
    ground-truth and predicted classes and instances) plus a bird's-eye view.
``projection``
    Share of points the range image keeps at each width (no labels needed).
``upsampling``
    Rebuilds the full-width range image from the half-width one by nearest,
    2D-linear and LPSAD's distance-aware upsampling, and measures the error at
    object edges, where 2D interpolation blends foreground and background
    (no labels needed).

    python tools/visualize.py scan -d /data/kitti/dataset -s 08 --index 0 \
        -p preds -o figures/scan.png
    python tools/visualize.py projection -d /data/kitti/dataset -s 08 -o figures/projection.png
    python tools/visualize.py upsampling -d /data/kitti/dataset -s 08 -o figures/upsampling.png

@ingroup tools
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpsad.config import LabelSpace, build_label_space, load_config  # noqa: E402
from lpsad.dataset import spherical_projection  # noqa: E402
from lpsad.model import TrilinearUpsample  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "lpsad-darknet53.yaml"
FOV_UP, FOV_DOWN, H = 3.0, -25.0, 64


# ----------------------------------------------------------------------- data
def scan_paths(root: Path, seq: str) -> list[Path]:
    """Sorted scan files of one sequence."""
    return sorted((root / "sequences" / seq / "velodyne").glob("*.bin"))


def read_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Points (N, 3) and remission (N,) of one ``.bin`` scan."""
    scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3], scan[:, 3]


def read_label(path: Path, labels: LabelSpace) -> tuple[np.ndarray, np.ndarray] | None:
    """Training class and instance id per point, or None if the file is missing."""
    if not path.exists():
        return None
    raw = np.fromfile(path, dtype=np.uint32).astype(np.int64)
    return labels.learning_lut[raw & 0xFFFF], raw >> 16


def project(values: np.ndarray, proj_idx: np.ndarray, fill=0) -> np.ndarray:
    """Per-point values -> (H, W) image through the projection's point index."""
    out = np.full(proj_idx.shape + values.shape[1:], fill, dtype=values.dtype)
    mask = proj_idx >= 0
    out[mask] = values[proj_idx[mask]]
    return out


# --------------------------------------------------------------------- colours
EMPTY = "#2b2b2b"  # pixels without a point


def cmap_with_empty(name: str):
    """Matplotlib colour map that draws NaN (empty pixels) in dark grey."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(EMPTY)
    return cmap

def class_colors(cfg: dict, labels: LabelSpace) -> np.ndarray:
    """RGB colour per training class from the SemanticKITTI colour map (BGR)."""
    cmap = cfg["label_cfg"]["color_map"]
    colors = np.array([cmap[int(labels.inverse_lut[c])][::-1] for c in range(labels.nclasses)],
                      dtype=np.float32) / 255.0
    colors[0] = 0.0  # unlabeled / empty: black
    return colors


def instance_colors(inst: np.ndarray, seed: int = 3) -> np.ndarray:
    """Distinct random colour per instance id; 0 (no instance) dark grey."""
    rng = np.random.default_rng(seed)
    palette = rng.uniform(0.25, 1.0, size=(int(inst.max()) + 1, 3))
    palette[0] = 0.15
    return palette[inst]


# --------------------------------------------------------------------- figures
def figure_scan(args, cfg, labels):
    """Range-image panels and a bird's-eye view of one scan."""
    root = Path(args.dataset)
    path = scan_paths(root, args.sequence)[args.index]
    xyz, remission = read_scan(path)
    px, py, depth, proj_idx = spherical_projection(xyz, H, args.width, FOV_UP, FOV_DOWN)
    mask = proj_idx >= 0
    colors = class_colors(cfg, labels)

    panels = [("range (m)", np.where(mask, project(depth, proj_idx), np.nan), "viridis_r"),
              ("remission", np.where(mask, project(remission, proj_idx), np.nan), "gray")]
    sources = [("ground truth", root / "sequences" / args.sequence / "labels")]
    if args.predictions:
        sources.append(("prediction", Path(args.predictions) / "sequences" / args.sequence
                        / "predictions"))
    bev = None
    for name, folder in sources:
        lab = read_label(folder / f"{path.stem}.label", labels)
        if lab is None:
            continue
        sem, inst = lab
        empty = np.array([int(EMPTY[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        sem_img = colors[project(sem, proj_idx)]
        sem_img[~mask] = empty
        panels.append((f"{name}: classes", sem_img, None))
        inst_img = instance_colors(project(np.where(labels.is_thing()[sem], inst, 0), proj_idx))
        inst_img[~mask] = empty
        panels.append((f"{name}: instances", inst_img, None))
        bev = (name, sem, inst)

    fig = plt.figure(figsize=(14, 1.15 * len(panels) + 5.5), constrained_layout=True)
    grid = fig.add_gridspec(len(panels) + 1, 1, height_ratios=[1] * len(panels) + [4.5])
    for i, (title, img, cmap) in enumerate(panels):
        ax = fig.add_subplot(grid[i])
        kw = {"cmap": cmap_with_empty(cmap)} if cmap else {}
        if title.startswith("range"):
            kw.update(vmin=0, vmax=50)
        ax.imshow(img, aspect="auto", interpolation="nearest", **kw)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(grid[-1])
    near = np.abs(xyz[:, :2]).max(1) < args.bev_range
    if bev is None:
        c = plt.cm.viridis(np.clip(xyz[near, 2] + 2, 0, 3) / 3)
        title = "bird's-eye view, coloured by height"
    else:
        name, sem, inst = bev
        thing = labels.is_thing()[sem]
        c = np.where(thing[:, None], instance_colors(np.where(thing, inst, 0)), colors[sem] * 0.6)
        c = c[near]
        title = f"bird's-eye view, {name}: things by instance, stuff dimmed by class"
    ax.scatter(xyz[near, 0], xyz[near, 1], s=0.15, c=c, linewidths=0)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.suptitle(f"sequence {args.sequence}, scan {path.stem}, 64 x {args.width}",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(args.out, dpi=args.dpi)
    return None


def coverage(xyz: np.ndarray, width: int) -> float:
    """Share of points that own a pixel at this width."""
    *_, proj_idx = spherical_projection(xyz, H, width, FOV_UP, FOV_DOWN)
    return float((proj_idx >= 0).sum()) / len(xyz)


def figure_projection(args, cfg, labels):
    """Points kept by the range image versus image width."""
    scans = scan_paths(Path(args.dataset), args.sequence)[: args.max_scans]
    rows = []
    for w in args.widths:
        vals = [coverage(read_scan(p)[0], w) for p in scans]
        rows.append((w, 100 * np.mean(vals), 100 * np.min(vals), 100 * np.max(vals)))

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    widths = [r[0] for r in rows]
    ax.plot(widths, [r[1] for r in rows], "o-", color="#3d6fb6")
    ax.fill_between(widths, [r[2] for r in rows], [r[3] for r in rows], color="#3d6fb6",
                    alpha=0.2, label="min-max over scans")
    for w, mean, *_ in rows:
        ax.annotate(f"{mean:.0f}%", (w, mean), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_ylim(0, 100)
    ax.set_xlabel("range image width (height 64)")
    ax.set_ylabel("points that own a pixel (%)")
    ax.set_title(f"Range projection keeps a fraction of each scan "
                 f"({len(scans)} scans, sequence {args.sequence})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(args.out, dpi=args.dpi)
    table = ["| Width | Points kept (mean) | min | max |", "|---|---|---|---|"]
    table += [f"| 64 x {w} | {m:.1f}% | {lo:.1f}% | {hi:.1f}% |" for w, m, lo, hi in rows]
    return "\n".join(table)


def upsample_all(rng: np.ndarray, xyz: np.ndarray, mask: np.ndarray):
    """Full-width range image rebuilt from every second column, three ways."""
    coarse, xyz_c, mask_c = rng[:, ::2], xyz[:, :, ::2], mask[:, ::2]
    wc = coarse.shape[1]
    j0 = np.arange(rng.shape[1]) // 2
    j1 = (j0 + 1) % wc
    nearest = coarse[:, j0]
    a, b = coarse[:, j0], coarse[:, j1]
    ma, mb = mask_c[:, j0], mask_c[:, j1]
    odd = (np.arange(rng.shape[1]) % 2 == 1)[None]
    both = 0.5 * (a + b)
    linear = np.where(odd & ma & mb, both, np.where(odd & mb & ~ma, b, a))
    up = TrilinearUpsample(rows=3, k=4)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x))[None]  # noqa: E731
    tri = up(t(coarse)[:, None].float(), t(xyz_c).float(), t(mask_c), t(xyz).float(),
             t(mask))[0, 0].numpy()
    return {"nearest": nearest, "2D linear": linear, "LPSAD trilinear": tri}


def figure_upsampling(args, cfg, labels):
    """Edge-preserving upsampling demo: errors of three methods at object edges."""
    scans = scan_paths(Path(args.dataset), args.sequence)[: args.max_scans]
    sums = {}
    example = None
    for k, path in enumerate(scans):
        xyz, _ = read_scan(path)
        *_, depth, proj_idx = spherical_projection(xyz, H, args.width, FOV_UP, FOV_DOWN)
        mask = proj_idx >= 0
        rng = project(depth, proj_idx)
        img_xyz = project(xyz, proj_idx).transpose(2, 0, 1)
        recon = upsample_all(rng, img_xyz, mask)
        # evaluate interpolated pixels (odd columns) whose coarse neighbours exist
        j0 = np.arange(rng.shape[1]) // 2
        j1 = (j0 + 1) % (rng.shape[1] // 2)
        mc = mask[:, ::2]
        ev = mask & mc[:, j0] & mc[:, j1] & (np.arange(rng.shape[1]) % 2 == 1)[None]
        edge = ev & (np.abs(rng[:, ::2][:, j0] - rng[:, ::2][:, j1]) > args.edge)
        for name, img in recon.items():
            err = np.abs(img - rng)
            s = sums.setdefault(name, np.zeros(4))
            s += (err[ev].sum(), ev.sum(), err[edge].sum(), edge.sum())
        if k == 0:
            example = (rng, recon, edge)

    rows = [(n, s[0] / s[1], s[2] / max(s[3], 1)) for n, s in sums.items()]
    rng, recon, edge = example
    if args.crop_start is None:
        # show the window where 2D interpolation smears most across edges
        bad = (np.abs(recon["2D linear"] - rng) > 1.0) & edge
        per_col = np.convolve(bad.sum(0), np.ones(args.crop_width), mode="valid")
        c0 = int(per_col.argmax())
    else:
        c0 = args.crop_start
    crop = slice(c0, c0 + args.crop_width)
    fig, axes = plt.subplots(len(recon) + 1, 2, figsize=(13, 1.0 * (len(recon) + 1) + 1.6),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [3, 1]})
    axes[0, 0].imshow(np.where(rng > 0, rng, np.nan)[:, crop], cmap=cmap_with_empty("viridis_r"),
                      vmin=0, vmax=50, aspect="auto", interpolation="nearest")
    axes[0, 0].set_title("ground truth: full-width range image, columns "
                         f"{c0}-{c0 + args.crop_width}", loc="left", fontsize=9)
    axes[0, 1].axis("off")
    for i, (name, img) in enumerate(recon.items(), start=1):
        err = np.where(rng > 0, np.abs(img - rng), np.nan)[:, crop]
        im = axes[i, 0].imshow(err, cmap=cmap_with_empty("Reds"), vmin=0, vmax=2.0,
                               aspect="auto", interpolation="nearest")
        mean, edge_err = next((r[1], r[2]) for r in rows if r[0] == name)
        axes[i, 0].set_title(f"{name}: |error| in metres (0-2 m)", loc="left", fontsize=9)
        axes[i, 1].barh([0, 1], [mean, edge_err], color=["#9aa7b5", "#c8553d"])
        axes[i, 1].set_yticks([0, 1], ["all", "edges"])
        axes[i, 1].set_xlim(0, max(r[2] for r in rows) * 1.15)
        axes[i, 1].set_title(f"mean |error|: {mean:.2f} m all, {edge_err:.2f} m edges",
                             fontsize=8)
    for ax in axes[:, 0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[1:, 0], shrink=0.6, label="|error| (m)")
    fig.suptitle(f"Rebuilding a 64 x {args.width} range image from 64 x {args.width // 2} "
                 f"({len(scans)} scans, sequence {args.sequence}; edges: neighbours "
                 f"differ by > {args.edge} m)", x=0.01, ha="left", fontsize=10)
    fig.savefig(args.out, dpi=args.dpi)
    table = ["| Upsampling | Mean abs. error, all (m) | at object edges (m) |",
             "|---|---|---|"]
    table += [f"| {n} | {a:.3f} | {e:.3f} |" for n, a, e in rows]
    return "\n".join(table)


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("@ingroup")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["scan", "projection", "upsampling"])
    ap.add_argument("-d", "--dataset", required=True, help="root with sequences/")
    ap.add_argument("-s", "--sequence", default="08")
    ap.add_argument("-o", "--out", required=True, help="output image (.png, .svg or .pdf)")
    ap.add_argument("-p", "--predictions", help="panoptic predictions root (scan figure)")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--index", type=int, default=0, help="scan index (scan figure)")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--widths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--max-scans", type=int, default=20)
    ap.add_argument("--bev-range", type=float, default=40.0, help="metres (scan figure)")
    ap.add_argument("--edge", type=float, default=1.0, help="edge threshold, metres")
    ap.add_argument("--crop-start", type=int, help="first column of the upsampling crop "
                    "(default: where 2D interpolation errs most)")
    ap.add_argument("--crop-width", type=int, default=400)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    args.sequence = f"{int(args.sequence):02d}"

    cfg = load_config(args.config)
    labels = build_label_space(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make = {"scan": figure_scan, "projection": figure_projection,
            "upsampling": figure_upsampling}[args.figure]
    table = make(args, cfg, labels)
    print(f"wrote {args.out}")
    if table:
        md = Path(args.out).with_suffix(".md")
        md.write_text(table + "\n")
        print(table)


if __name__ == "__main__":
    main()
