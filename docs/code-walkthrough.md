# Code walkthrough: one scan through LPSAD

This page follows a single SemanticKITTI scan from disk to panoptic labels and
points to the function doing each step. Read it next to the code. For the
design and its reasons see [architecture](architecture.md) and
[design decisions](design-decisions.md); for the API, the Doxygen reference
(`make -C docs html`), grouped by the same subsystems.

All shapes are for the default configuration: Darknet53, a 64 x 2048 range
image, batch size B. Paths are relative to `train/tasks/panoptic/`.

## 0. Map

| Step | Function | File |
|---|---|---|
| Labels and config | `load_config`, `build_label_space` | `lpsad/config.py` |
| Scan → tensors | `SemanticKittiPanoptic.__getitem__` | `lpsad/dataset.py` |
| Network | `LPSAD.forward` | `lpsad/model.py` |
| Losses | `LPSADLoss.forward` | `lpsad/losses.py` |
| Training loop | `Trainer.train_epoch`, `Trainer.validate` | `lpsad/trainer.py` |
| Instances | `extract_instances`, `panoptic_inference` | `lpsad/cluster.py` |
| Metrics | `PanopticEval` | `lpsad/metrics.py` |
| Entry points | `train.py`, `infer.py`, `evaluate.py`, `tools/*.py` | |

## 1. Labels: from raw ids to classes, things and categories

`load_config` reads `config/lpsad-darknet53.yaml` and the label file it
points to (the semantic task's `semantic-kitti.yaml`). `build_label_space`
turns them into look-up tables (`LabelSpace`):

- `learning_lut`: raw SemanticKITTI id (e.g. 10 "car", 252 "moving-car") →
  training class 0..19 (both cars → 1). A `.label` file stores
  `semantic | instance << 16` per point, so `learning_lut[raw & 0xFFFF]` gives
  the class and `raw >> 16` the instance id.
- `inverse_lut`: training class → raw id, used when writing predictions.
- `things`: classes with instances (1..8: car ... motorcyclist).
- `class_to_category`: class → vehicle / human / ground / structure / object /
  nature, for the category head.
- `content`: point frequency per class, for the class weights in the loss.

## 2. One scan becomes tensors

`SemanticKittiPanoptic.__getitem__(i)`:

1. **Read.** `velodyne/NNNNNN.bin` → `xyz` (N, 3) and `remission` (N,), with
   N ≈ 120 000. With ground truth, `labels/NNNNNN.label` → `sem` (N,).
2. **Augment** (training only, `Augment`): random yaw rotation, mirroring and
   scaling of `xyz`, before projection, so occlusion stays physically right.
3. **Instance targets** (`instance_targets`): thing points with a non-zero
   instance id get compact ids 1..K. Each instance's centre is the mean of all
   its points in the full cloud, and each point's target offset is
   `x - centre`, in metres.
4. **Project** (`spherical_projection`, paper Eq. 1). Yaw maps to a column and
   pitch to a row:

   ```
   u = 0.5 * (1 - atan2(y, x) / pi) * W
   v = (1 - (asin(z / r) + |fov_down|) / fov) * H
   ```

   Points are written from far to near, so each pixel ends up with its
   closest point. `proj_idx` (H, W) records which point owns each pixel
   (-1 = empty). Only about 78% of points own a pixel at 2048 width, 41% at
   1024 (see [figures](figures/README.md)).
5. **Assemble.** Gathering per-point arrays through `proj_idx` gives the
   images:

   | Key | Shape | Meaning |
   |---|---|---|
   | `input` | (5, 64, 2048) | range, x, y, z, remission, normalised; 0 on empty pixels |
   | `xyz` | (3, 64, 2048) | raw 3D point per pixel (for upsampling and clustering) |
   | `mask` | (64, 2048) | pixel has a point |
   | `sem`, `cat`, `inst` | (64, 2048) | class, category, instance id (0 = none) |
   | `offset` | (3, 64, 2048) | target offset to the instance centre |
   | `px`, `py` | (N,) | pixel of every point, including hidden ones |
   | `point_sem`, `point_inst` | (N,) | per-point ground truth, for evaluation |

   `collate` stacks the images into a batch and keeps per-point data as
   lists, since N differs between scans.

## 3. The network

`LPSAD.forward(image, xyz, mask)`:

**Encoder** (`DarknetBackbone`, the RangeNet++ Darknet53, 40.6 M of the
model's 73.5 M parameters). Each stage halves only the width:

| Output stride | Skip stored | Shape (B = 1) |
|---|---|---|
| 1 | after the stem | (1, 32, 64, 2048) |
| 2 | after stage 1 | (1, 64, 64, 1024) |
| 4 | after stage 2 | (1, 128, 64, 512) |
| 8 | after stage 3 | (1, 256, 64, 256) |
| 16 | after stage 4 | (1, 512, 64, 128) |
| 32 | encoder output | (1, 1024, 64, 64) |

`run_layer` stores the input of every downsampling layer as a skip. Unlike
upstream it doesn't `detach()` it, so the losses train the early layers
through the skips too.

**Point cloud pyramid** (`point_pyramid`): `xyz[..., ::2]` once per stride.
A 3x3 convolution with stride 2 and padding 1 centres output column j on input
column 2j, so this is the 3D point at the centre of each kernel: (3, 64, 1024),
(3, 64, 512), ... (3, 64, 64).

**Decoders** (`Decoder`, 5 x `DecoderStage`), one semantic and one instance,
with identical structure and separate weights:

```
(1024, 64, 64)  --up x2 + skip 512--> (512, 64, 128)
                --up x2 + skip 256--> (256, 64, 256)
                --up x2 + skip 128--> (128, 64, 512)
                --up x2 + skip 64 --> ( 64, 64, 1024)
                --up x2 + skip 32 --> ( 32, 64, 2048)
```

Each stage: `TrilinearUpsample` → concatenate the skip → 3x3 conv, BN,
LeakyReLU → Darknet residual block.

**`TrilinearUpsample`, the paper's key idea.** Take fine pixel (h, w) with 3D
point `x_f`:

1. `_candidates`: the two coarse columns around it, `j0 = w // 2` and
   `j1 = j0 + 1` (wrapping at 360°), in rows h-1, h and h+1: 6 candidates.
2. `weights`: 3D distance from `x_f` to each candidate's point (`inf` if
   the coarse pixel is empty), keep the 4 closest, weight `1 / (d + 0.001)`,
   normalise. Example: a car pixel whose candidates are 0.1 m (car), 0.2 m
   (car) and 6 m, 7 m (wall behind) away gets weights ≈ 0.65, 0.33, 0.01, 0.01.
   2D interpolation would give the wall half the weight.
3. `forward`: the feature is the weighted sum of the candidates' features.
   Weights depend only on geometry, so there is no gradient through them and
   nothing to learn.

**Heads.** The semantic decoder output (32 channels) → `embedding` (32) →
`class_head` (20) and `category_head` (7), plus `embedding_err` (1). The
instance decoder output → `offset_head` (3, metres) and `offset_err` (1).
`ErrorHead` ends in a softplus, so radii are positive.

## 4. Losses

`LPSADLoss.forward(out, batch)` works on the thing pixels of the batch.
Instances from different scans must not mix, so each gets a global id
`scan * (max_id + 1) + inst`, made compact with `torch.unique`.

The per-instance averages of the paper (`1/I Σ_i 1/P_i Σ_p ...`) come from
`segment_mean(values, idx, n)`: sum per instance with `index_add_`, divide
by the point count, then take the mean over instances.

| Paper | Term | Code |
|---|---|---|
| Eq. 2 | offset regression | `terms["center"]`: squared error of `offset` vs target, per-instance mean |
| Eq. 3 | offset radius | `terms["center_err"]`: `offset_err` regresses the detached offset error |
| Eq. 4 | classes | `weighted_ce` with `w_c = 1 / ln(1.02 + f_c)` |
| (Sec. III-C) | categories | `weighted_ce` on `cat_logits` |
| Eq. 5 | attract | `terms["attract"]`: distance of each embedding to its instance mean |
| Eq. 6 | repel | `terms["repel"]`: `1 / (d² + delta)` between instance means of the same scan |
| Eq. 7 | embedding radius | `terms["embed_err"]` |

The total is the weighted sum (weights in `config: loss`). `stats` returns
the mean offset and embedding errors, which set the radius scale.

## 5. Training loop

`Trainer.train_epoch`, per batch:

1. Forward under `autocast` (bf16 on Ampere GPUs, fp16 with `GradScaler`
   otherwise, nothing on CPU).
2. `loss / accumulate` → backward. Every `accumulate` batches: unscale, clip
   gradients, Adam step, scheduler step. The effective batch is
   `batch_size * accumulate`.
3. Learning rate (`lr_lambda`): linear warm-up over `warmup_epochs`, then
   `lr * 0.99 ** epochs_since_warmup`, as in the paper.

`Trainer.validate` runs the model and the loss, clusters with the current
radius scale, and adds every scan to `PanopticEval`. It then updates the
radius scale to `val error / train error`, so the scale lags one epoch.
`fit` saves `last.pt` every epoch and `best.pt` when validation PQ improves.

## 6. From outputs to instances

`panoptic_inference` per scan:

1. `sem = argmax(logits)` per pixel.
2. `extract_instances` over the pixels whose predicted class is a thing:

   ```
   remaining = all thing pixels
   while remaining:
       s = random pixel in remaining
       cand = remaining with the same class as s
       join = cand whose |centre - centre[s]| <= radius   (and/or embeddings)
       give them a new id, remove them from remaining
   ```

   `centre = xyz - offset`. The radius is the seed's learned radius times
   the radius scale (`radius_mode: seed`, paper), or seed plus candidate
   (`sum`). `remaining` shrinks every iteration, so the loop runs about once
   per instance, plus once per stray pixel.
3. Unproject: every point takes the labels of its pixel, `sem[py, px]` and
   `inst[py, px]` (paper Sec. III-E). This is why projection loss matters.

`infer.py` writes `inverse_lut[sem] | inst << 16` as `uint32`.

## 7. Evaluation

`PanopticEval.add(pred_sem, pred_inst, gt_sem, gt_inst)` per scan, per class:

1. Drop points whose ground truth is an ignored class.
2. Shift instance ids by one, so stuff (instance 0) forms one segment per
   class.
3. Count every (prediction, ground truth) segment pair in one `np.unique` over
   `pred + 2**32 * gt`. Pairs with IoU > 0.5 are matches (true positives, at
   most one per segment).
4. Unmatched segments with at least 50 points count as false negatives or
   false positives.

`results()` computes per class `SQ = Σ IoU / TP`, `RQ = TP / (TP + FP/2 +
FN/2)` and `PQ = SQ × RQ`, then averages over all, thing and stuff classes.
PQ† uses IoU in place of PQ for stuff.

## 8. The training-free tools

The `tools/` scripts reuse these pieces without a trained model:

- `baseline.py`: any semantic segmentation → `lpsad/geometric.py`
  (`euclidean_clusters`: k-d tree pairs within a radius → connected
  components per thing class) → `PanopticEval`.
- `upper_bounds.py`: the dataset's own targets, plus noise, go through
  `extract_instances` and `PanopticEval`.
- `benchmark.py`: random weights through `LPSAD` and `LPSADLoss`, timed.
- `visualize.py`, `report.py`: figures and tables.

## Where to change things

| To change | Edit |
|---|---|
| Image size, sensor field of view | `config: sensor` |
| Augmentation | `Augment` in `lpsad/dataset.py`, `config: train.augment` |
| Upsampling neighbourhood | `TrilinearUpsample` (`trilinear_rows`, `trilinear_k`) |
| Decoder widths | `config: model.decoder.channels` |
| Loss weights | `config: loss` |
| Clustering rule | `extract_instances`, `config: cluster` |
| A new head | see [development.md](development.md#adding-a-model-or-head) |
