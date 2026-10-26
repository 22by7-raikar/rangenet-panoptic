#!/usr/bin/env python3
"""Speed and memory of LPSAD on this machine (no training, no dataset needed).

1. **Inference**: network forward time per scan at each width (batch 1).
2. **Clustering**: instance extraction time per scan, from true centre
   offsets plus noise (needs ``--dataset`` with labels; skipped otherwise).
3. **Training memory**: peak GPU memory of one optimisation step for each
   width and batch size, or "OOM". Answers "what fits on my GPU?".

Weights are random: speed and memory don't depend on them. Without
``--dataset``, synthetic scans stand in for real ones.

    python tools/benchmark.py --widths 512 1024 2048 --batch-sizes 1 2 4 --amp bf16
    python tools/benchmark.py -d /data/semantic-kitti --widths 1024 --json bench.json
"""

import argparse
import copy
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
from lpsad.cluster import extract_instances  # noqa: E402
from lpsad.config import build_label_space, load_config  # noqa: E402
from lpsad.dataset import SemanticKittiPanoptic, collate  # noqa: E402
from lpsad.losses import LPSADLoss  # noqa: E402
from lpsad.model import LPSAD  # noqa: E402
from lpsad.trainer import autocast_context, to_device  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(1000 * (time.perf_counter() - start))
    return round(statistics.median(times), 1)


def dataset(cfg, labels, args, width, root):
    cfg = copy.deepcopy(cfg)
    cfg["sensor"]["width"] = width
    return cfg, SemanticKittiPanoptic(root, args.sequences, cfg, labels, gt=True,
                                      max_scans=max(args.iters, max(args.batch_sizes)))


def bench_width(cfg, labels, args, width, root, device):
    cfg, ds = dataset(cfg, labels, args, width, root)
    model = LPSAD(cfg, labels.nclasses, labels.ncategories).to(device)
    result = {"width": width}

    # 1. inference
    model.eval()
    batch = to_device(collate([ds[0]]), device)
    with torch.no_grad():
        def forward():
            with autocast_context(args.amp, device):
                model(batch["input"], batch["xyz"], batch["mask"])
        result["network_ms"] = timed(forward, device, args.warmup, args.iters)

    # 2. clustering on (noisy) true offsets
    if args.dataset:
        is_thing = torch.from_numpy(labels.is_thing()).to(device)
        gen = torch.Generator().manual_seed(0)
        times = []
        for i in range(min(args.iters, len(ds))):
            item = to_device(collate([ds[i]]), device)
            noise = torch.randn(item["offset"][0].shape, generator=gen).to(device) * 0.2
            err = noise.norm(dim=0, keepdim=True)
            sync(device)
            start = time.perf_counter()
            extract_instances(item["sem"][0], item["mask"][0], is_thing, item["xyz"][0],
                              item["offset"][0] + noise, err, torch.zeros_like(err), err,
                              features="center", radius_mode=args.radius_mode, generator=gen)
            sync(device)
            times.append(1000 * (time.perf_counter() - start))
        result["clustering_ms"] = round(statistics.median(times), 1)

    # 3. training memory per batch size
    if device.type == "cuda":
        model.train()
        criterion = LPSADLoss(cfg, labels).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
        result["train_peak_gb"] = {}
        for bs in args.batch_sizes:
            batch = to_device(collate([ds[i % len(ds)] for i in range(bs)]), device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                def step(batch=batch):
                    opt.zero_grad(set_to_none=True)
                    with autocast_context(args.amp, device):
                        out = model(batch["input"], batch["xyz"], batch["mask"])
                    loss = criterion(out, batch)[0]
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                ms = timed(step, device, 1, 3)
                peak = torch.cuda.max_memory_allocated(device) / 2**30
                result["train_peak_gb"][bs] = {"peak_gb": round(peak, 2), "step_ms": ms}
            except torch.cuda.OutOfMemoryError:
                result["train_peak_gb"][bs] = "OOM"
            torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", help="SemanticKITTI root with labels (optional)")
    ap.add_argument("-s", "--sequences", nargs="+", type=int, default=[8])
    ap.add_argument("-c", "--config", default=str(HERE / "config" / "lpsad-darknet53.yaml"))
    ap.add_argument("--layers", type=int, choices=[21, 53], default=53)
    ap.add_argument("--widths", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--radius-mode", choices=["seed", "sum"], default="seed")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    cfg = load_config(args.config)
    cfg["model"]["backbone"]["layers"] = args.layers
    labels = build_label_space(cfg)
    info = {"device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "amp": args.amp, "darknet": args.layers}
    print(json.dumps(info))

    with tempfile.TemporaryDirectory() as tmp:
        root = args.dataset
        if root is None:
            from synthetic import write_dataset
            root = write_dataset(Path(tmp), sequences=tuple(args.sequences),
                                 scans_per_seq=max(args.iters, max(args.batch_sizes)))
        results = []
        for w in args.widths:
            results.append(bench_width(cfg, labels, args, w, root, device))
            print(json.dumps(results[-1]), flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps({"info": info, "results": results}, indent=2))


if __name__ == "__main__":
    main()
