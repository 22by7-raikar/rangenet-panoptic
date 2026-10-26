# LiDAR-Bonnetal Panoptic Segmentation (LPSAD)

A re-implementation of

> A. Milioto, J. Behley, C. McCool, C. Stachniss. *LiDAR Panoptic Segmentation
> for Autonomous Driving.* IROS 2020.

on the RangeNet++ Darknet backbones of this repository. The paper's code was
never released; this follows the paper's equations and ablations on a
modern PyTorch 2.x training stack.

Design documentation: [architecture](../../../docs/architecture.md) (diagrams of
every subsystem) and [design decisions](../../../docs/design-decisions.md).

## Method

| Paper | Here |
|---|---|
| Range-image projection, 5 x H x W input (Sec. III-A) | `lpsad/dataset.py` |
| Shared Darknet53 encoder + point cloud pyramid (III-B) | `lpsad/model.py` (`DarknetBackbone`, `point_pyramid`) |
| Distance-aware trilinear upsampling (III-C, Fig. 4) | `TrilinearUpsample` |
| Instance decoder: 3D centre offsets + learned radius (Eqs. 2-3) | `offset_head`, `offset_err` |
| Semantic decoder: 32-d embedding, class + category heads (Eq. 4) | `embedding`, `class_head`, `category_head` |
| Attract / repel embedding losses + embedding radius (Eqs. 5-7) | `lpsad/losses.py` |
| Iterative seed-based instance extractor (III-D) | `lpsad/cluster.py` |
| Labels for all points via (u, v) look-up (III-E) | `panoptic_inference` |
| Learned radii scaled by val/train error ratio | `Trainer.validate` (`radius_scale: auto`) |
| Adam 1e-4, 1 epoch warm-up, x0.99/epoch, 200 epochs | `config/lpsad-darknet53.yaml` |

Where the paper leaves details open:

- **Upsampling neighbours.** The encoder only downsamples horizontally, so
  candidates are the two coarse columns bracketing a fine pixel in 3 rows;
  the 4 closest in 3D are blended by inverse distance (`trilinear_rows`,
  `trilinear_k`). The range image wraps around at 360 degrees.
- **Decoder stage.** Upsample, concatenate the skip, conv-BN-LeakyReLU,
  Darknet residual block (Fig. 2 shows concatenation).
- **Class weights.** Eq. 4 writes `1/log(f + eps)`, which is negative for
  `f + eps < 1`; this uses the ENet form `1/ln(1.02 + f)`.
- **Repel loss.** Eq. 6 is `1/d^2`, unbounded while embeddings are still
  collapsed; this uses `1/(d^2 + delta)` with `repel_delta: 1.0`.
  Pairs are formed within each scan.
- **Instance centres** are the mean of all points of the instance in the
  full cloud, not only the projected ones.
- **Clustering radius** (`cluster.radius_mode`, an option, not a deviation).
  The default `seed` follows the paper: a candidate joins if it lies within
  the seed's learned radius. `sum` (seed radius + candidate radius) is offered
  for noisy offsets, where `seed` fragments objects; which works better on
  real data is measured with `tools/upper_bounds.py` rather than assumed.

## Modernisation compared to the semantic task

- PyTorch 2.x, Python 3.10+, no TensorFlow, no `imp` / `__init__` path hacks.
- Mixed precision (`amp: bf16` for Ampere GPUs such as the RTX 30 series,
  `fp16` with a gradient scaler for older ones), optional `torch.compile`.
- Gradient accumulation (`accumulate`) for small GPUs.
- Skip connections are not detached (RangeNet++ detached them, so the
  encoder only learned through its bottleneck).
- Backbone can be initialised from a RangeNet++ semantic model
  (`-p /path/to/darknet53`), optionally frozen for the first epochs and with
  frozen BatchNorm statistics for small batches.
- Global augmentation (yaw rotation, mirroring, scaling) before projection.
- Fixed `proj_mask` (the semantic code drops the point with index 0).
- Resumable checkpoints (`last.pt`, `best.pt` by validation PQ),
  TensorBoard logging when `tensorboard` is installed.
- Point-level PQ / PQ-dagger / RQ / SQ / mIoU evaluation following
  semantic-kitti-api, plus unit and end-to-end tests.

## Usage

```sh
cd train/tasks/panoptic
pip install -r requirements.txt

# train (optionally from RangeNet++ weights, see the top-level README)
python train.py -c config/lpsad-darknet53.yaml -d /data/semantic-kitti \
    -l logs/lpsad -p /models/darknet53 \
    --set train.freeze_backbone_epochs=10 sensor.width=1024

# resume
python train.py -c config/lpsad-darknet53.yaml -d /data/semantic-kitti \
    -l logs/lpsad --resume logs/lpsad/last.pt

# predict (semantic-kitti-api format) and evaluate
python infer.py -m logs/lpsad/best.pt -d /data/semantic-kitti -o preds --split valid
python evaluate.py -d /data/semantic-kitti -p preds --split valid

# tests (synthetic data, CPU)
python -m pytest tests
```

`--set key.sub=value` overrides any config value.

### On a 6 GB laptop GPU (e.g. RTX 3060 laptop)

Start with `sensor.width=1024`, `train.batch_size=2`, `train.accumulate=4`,
`train.amp=bf16`, a pretrained backbone with `freeze_backbone_epochs` > 0 and
`freeze_backbone_bn: true`. Use width 512 only for debugging: thing classes
degrade badly at that resolution (RangeNet++ Table I). Keep `workers` at or
below the number of physical cores.

## Real numbers without training

`tools/` measures things on real SemanticKITTI data without training
anything, from as little as 1 GB of data. Results go to JSON with `--json`.

**1. Data: one sequence, or a subset of one.** The KITTI archive is 85 GB,
but single scans can be read out of it remotely, together with their labels:

```sh
pip install remotezip
python tools/fetch_sequences.py -o /data/kitti -s 08 --max-scans 500   # ~1 GB
# dataset root for every other command: /data/kitti/dataset
```

Sequence sizes (velodyne): 00 8.8 GB, 01 1.9, 02 9.4, 03 1.6, 04 0.55,
05 5.5, 06 2.2, 07 2.1, 08 (validation) 8.0, 09 3.2, 10 2.4. KITTI is for
non-commercial use: accept its terms at
[cvlibs.net](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) first.

**2. Projection and clustering ceilings** (ground truth only, CPU):

```sh
python tools/upper_bounds.py -d /data/kitti/dataset -s 8 --max-scans 200 --json bounds.json
```

For each image width: points kept by the projection, thing instances that
nearly vanish, and the PQ a perfect network would reach, which is the
ceiling of the range-image approach. Then clustering with true centre offsets
plus 0 / 0.1 / 0.2 / 0.4 m of noise under the paper's radius rule (`seed`),
the `sum` option and a fixed radius: how accurate the offsets must be, and
which rule to use.

**3. Training-free panoptic baseline:** RangeNet++ semantics + Euclidean
clustering. First get semantic predictions with a pre-trained RangeNet++ (see
the top-level README for model links). This also checks that the 2019
checkpoints load:

```sh
cd ../semantic && python infer.py -d /data/kitti/dataset -l /data/rangenet_preds -m /models/darknet53
cd ../panoptic
python tools/baseline.py -d /data/kitti/dataset -p /data/rangenet_preds -s 08 --json baseline.json
python tools/baseline.py -d /data/kitti/dataset --semantics gt -s 08   # clustering alone
```

The paper's comparable number is its two-stage baseline, RangeNet++ +
PointPillars (37.1 PQ on the test set, Table I).

**4. Speed and memory on your GPU** (random weights, no dataset needed):

```sh
python tools/benchmark.py --widths 512 1024 2048 --batch-sizes 1 2 4 --amp bf16 --json bench.json
```

Network and clustering milliseconds per scan, and the peak memory of a
training step per width and batch size: what fits on a 6 GB GPU. Use
`--amp fp16` on GPUs without bf16 (e.g. T4).

**5. Training on a subset**, when you get there: any sequences can be
selected, e.g. the small 03, 04 and 07 (4.3 GB) with 200 validation scans:

```sh
python train.py -d /data/kitti/dataset -l logs/subset -p /models/darknet53 \
    --set train.split.train=[3,4,7] train.val_max_scans=200 sensor.width=1024
```

## Results of this implementation

Sequence 08 (validation). Fill in as runs complete; record the command and
commit.

| Run | Scans | PQ | PQ-th | PQ-st | mIoU | Notes |
|---|---|---|---|---|---|---|
| Projection ceiling, 64 x 1024 | | | | | | `upper_bounds.py` |
| RangeNet++ (darknet53) + clustering | | | | | | `baseline.py` |
| True semantics + clustering | | | | | | `baseline.py --semantics gt` |
| LPSAD, trained on a subset | | | | | | `train.py` |

## Reference results (paper, SemanticKITTI)

| | PQ | PQ-dagger | RQ | SQ | PQ-th | PQ-st | mIoU | FPS |
|---|---|---|---|---|---|---|---|---|
| Test, with category loss (Table I) | 38.0 | 47.0 | 48.2 | 76.5 | 25.6 | 47.1 | 50.9 | 11.8 |
| Val, trilinear, all clustering features (Tables III/IV) | 36.5 | 46.1 | | | | | 50.7 | 11.8 |

Validation ablations (Table IV) give targets per component: centres with a
fixed radius 34.4 PQ, + learned radius 35.8, + embeddings 36.5.
