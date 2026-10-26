#!/usr/bin/env python3
"""Tables and charts from the JSON written by the other tools.

Reads any mix of ``upper_bounds.py``, ``baseline.py`` and ``benchmark.py``
outputs (``--json``), recognised by their content, and writes
``OUT/report.md`` (Markdown tables for the README or slides) plus one PNG
chart per result type.

    python tools/report.py bounds.json baseline.json oracle.json bench.json -o report \
        --names "Upper bounds" "RangeNet++ + clustering" "True semantics + clustering" "Laptop"

@ingroup tools
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#3d6fb6", "#c8553d", "#4c956c", "#b08900", "#7b5ea7"]
METRICS = ("PQ", "PQ_th", "PQ_st", "RQ", "SQ", "mIoU")


def kind(data: dict) -> str:
    """Which tool wrote this JSON: 'bounds', 'baseline' or 'benchmark'."""
    if "metrics" in data:
        return "baseline"
    if "info" in data:
        return "benchmark"
    if data.get("results") and "projection_ceiling" in data["results"][0]:
        return "bounds"
    raise ValueError("not a JSON written by upper_bounds.py, baseline.py or benchmark.py")


def table(header: list[str], rows: list[list]) -> str:
    """Markdown table from a header and rows of cells."""
    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def report_baselines(items: list[tuple[str, dict]]) -> str:
    """One table row per baseline run."""
    rows = [[name] + [d["metrics"].get(m, "") for m in METRICS]
            + [d["args"].get("radius", "")] for name, d in items]
    return "### Panoptic quality (%)\n\n" + table(["Run", *METRICS, "Radius (m)"], rows)


def report_bounds(name: str, data: dict, out: Path) -> str:
    """Projection and clustering ceilings: two tables and two charts."""
    res = data["results"]
    lost_key = next(k for k in res[0] if k.startswith("instances_below_"))
    rows = [[f"64 x {r['width']}", r["points_visible_pct"], r[lost_key],
             *[r["projection_ceiling"][m] for m in ("PQ", "PQ_th", "PQ_st", "mIoU")]]
            for r in res]
    md = [f"### {name}: projection ceiling ({res[0]['scans']} scans)\n",
          table(["Width", "Points kept (%)", "Instances nearly lost (%)",
                 "PQ ceiling", "PQ-th", "PQ-st", "mIoU"], rows)]

    keys = list(res[-1]["clustering"])
    noises = sorted({float(k.split()[1]) for k in keys})
    modes = list(dict.fromkeys(k.split(" / ")[1] for k in keys))
    for r in res:
        rows = [[f"{n} m"] + [r["clustering"][f"noise {n} m / {m}"]["PQ_th"] for m in modes]
                for n in noises]
        md += [f"\n### {name}: clustering with true offsets + noise, 64 x {r['width']} "
               f"(PQ-th; ceiling {r['projection_ceiling']['PQ_th']:.1f})\n",
               table(["Offset noise", *modes], rows)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    widths = [r["width"] for r in res]
    for i, m in enumerate(("PQ", "PQ_th", "PQ_st")):
        axes[0].plot(widths, [r["projection_ceiling"][m] for r in res], "o-",
                     color=COLORS[i], label=m.replace("_", "-"))
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(widths, [str(w) for w in widths])
    axes[0].set(xlabel="range image width", ylabel="ceiling (%)",
                title="What a perfect range-image network could reach")
    r = res[-1]
    for i, m in enumerate(modes):
        axes[1].plot(noises, [r["clustering"][f"noise {n} m / {m}"]["PQ_th"] for n in noises],
                     "o-", color=COLORS[i], label=m)
    axes[1].axhline(r["projection_ceiling"]["PQ_th"], color="grey", ls="--", lw=1,
                    label="projection ceiling")
    axes[1].set(xlabel="offset noise per axis (m)", ylabel="PQ-th (%)",
                title=f"Clustering vs offset accuracy (64 x {r['width']})")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    path = out / "upper_bounds.png"
    fig.savefig(path, dpi=150)
    return "\n".join(md) + f"\n\n![{name}]({path.name})"


def report_benchmark(name: str, data: dict, out: Path) -> str:
    """Latency and training-memory table and chart."""
    info, res = data["info"], data["results"]
    batch_sizes = sorted({int(b) for r in res for b in r.get("train_peak_gb", {})})
    header = ["Width", "Network (ms)", "Clustering (ms)"] + [f"Train peak, batch {b} (GB)"
                                                              for b in batch_sizes]
    rows = []
    for r in res:
        mem = r.get("train_peak_gb", {})
        cells = []
        for b in batch_sizes:
            v = mem.get(str(b), mem.get(b, "-"))
            cells.append(v["peak_gb"] if isinstance(v, dict) else v)
        rows.append([f"64 x {r['width']}", r["network_ms"], r.get("clustering_ms", "-"), *cells])
    md = (f"### {name}: {info['device']}, {info['amp']}, Darknet{info['darknet']}\n\n"
          + table(header, rows))

    fig, axes = plt.subplots(1, 2 if batch_sizes else 1, figsize=(11, 3.6),
                             constrained_layout=True, squeeze=False)
    widths = [r["width"] for r in res]
    labels = [str(w) for w in widths]
    axes[0, 0].bar(labels, [r["network_ms"] for r in res], color=COLORS[0], label="network")
    if all("clustering_ms" in r for r in res):
        axes[0, 0].bar(labels, [r["clustering_ms"] for r in res],
                       bottom=[r["network_ms"] for r in res], color=COLORS[1],
                       label="clustering")
    axes[0, 0].axhline(100, color="grey", ls="--", lw=1, label="10 Hz sensor")
    axes[0, 0].set(xlabel="range image width", ylabel="ms per scan",
                   title=f"Inference time ({info['device']})")
    axes[0, 0].legend(fontsize=8)
    if batch_sizes:
        ax = axes[0, 1]
        for i, r in enumerate(res):
            mem = r["train_peak_gb"]
            pts = [(b, mem.get(str(b), mem.get(b))) for b in batch_sizes]
            pts = [(b, v["peak_gb"]) for b, v in pts if isinstance(v, dict)]
            if pts:
                ax.plot(*zip(*pts, strict=True), "o-", color=COLORS[i % len(COLORS)],
                        label=f"64 x {r['width']}")
        ax.axhline(6, color="grey", ls="--", lw=1, label="6 GB")
        ax.set(xlabel="batch size", ylabel="peak memory (GB)", title="Training step memory")
        ax.set_xticks(batch_sizes)
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    path = out / "benchmark.png"
    fig.savefig(path, dpi=150)
    return md + f"\n\n![{name}]({path.name})"


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("@ingroup")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="JSON files from the other tools")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--names", nargs="*", help="display name per input (default: file name)")
    args = ap.parse_args()
    names = args.names or [Path(p).stem for p in args.inputs]
    if len(names) != len(args.inputs):
        ap.error("give one name per input")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loaded = [(n, json.loads(Path(p).read_text())) for n, p in zip(names, args.inputs,
                                                                   strict=True)]
    sections = []
    baselines = [(n, d) for n, d in loaded if kind(d) == "baseline"]
    if baselines:
        sections.append(report_baselines(baselines))
    for n, d in loaded:
        if kind(d) == "bounds":
            sections.append(report_bounds(n, d, out))
        elif kind(d) == "benchmark":
            sections.append(report_benchmark(n, d, out))
    text = "\n\n".join(sections) + "\n"
    (out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
