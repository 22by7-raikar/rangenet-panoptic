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
- **Clustering radius** (`cluster.radius_mode`). The paper compares a
  candidate against the seed's learned radius (`seed`). Two noisy centre
  predictions with errors e1, e2 can be up to e1 + e2 apart, and in our
  tests `seed` fragmented objects into many small clusters, so the default
  is `sum`: seed radius + candidate radius. Set `seed` to follow the paper.

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

## Reference results (paper, SemanticKITTI)

| | PQ | PQ-dagger | RQ | SQ | PQ-th | PQ-st | mIoU | FPS |
|---|---|---|---|---|---|---|---|---|
| Test, with category loss (Table I) | 38.0 | 47.0 | 48.2 | 76.5 | 25.6 | 47.1 | 50.9 | 11.8 |
| Val, trilinear, all clustering features (Tables III/IV) | 36.5 | 46.1 | | | | | 50.7 | 11.8 |

Validation ablations (Table IV) give targets per component: centres with a
fixed radius 34.4 PQ, + learned radius 35.8, + embeddings 36.5.
