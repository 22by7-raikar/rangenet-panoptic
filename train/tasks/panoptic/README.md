# Training-Free LiDAR Panoptic Segmentation

This package extends a pretrained RangeNet++ semantic model into a panoptic
LiDAR pipeline without training an instance network. It operates on the
existing spherical projection and retains the original point ordering.

## Method

1. Run RangeNet++ to obtain a semantic class distribution for each range-image
   pixel.
2. Map semantic labels and confidence back to the original LiDAR points.
3. For SemanticKITTI *thing* classes, connect compatible neighbouring range
   pixels using class-aware, range-adaptive 3D thresholds.
4. Use connected components to allocate instance identifiers; stuff classes
   keep instance id zero.
5. Pack the result into the SemanticKITTI format: `(instance_id << 16) |
   semantic_id`.

The horizontal dimension wraps across the 360-degree LiDAR seam. Projection
collisions follow the existing RangeNet++ convention: the nearest point wins,
and its instance is transferred back to other points landing on the pixel.

## Current status

`postproc.py` contains the reusable connected-component implementation.
`infer.py` invokes a pretrained semantic model and writes packed `.label`
files. It deliberately refuses an existing output directory so prior inference
results are not destroyed accidentally.

## Instance-extraction ablation

Two inference-time methods are included:

- `range-image` (default): the main class-aware, confidence-aware,
  range-adaptive connected-component method.
- `euclidean`: a fixed-radius 3D spatial-hash baseline. It intentionally does
  not use range-image connectivity or range scaling.

Use `--instance-method euclidean` to write a baseline prediction set, then
evaluate both output directories with the official SemanticKITTI evaluator.

## Visual inspection

Export an input scan and packed prediction as an ASCII PLY. Stuff is colored
by semantic class; thing instances receive deterministic color tints so merges
and splits are easy to see in CloudCompare or MeshLab.

```bash
python3 tasks/panoptic/visualize.py \
  --scan /path/to/000000.bin \
  --prediction /path/to/000000.label \
  --labels tasks/semantic/config/labels/semantic-kitti.yaml \
  --output /path/to/000000-panoptic.ply
```

Run the unit tests from the `train` directory:

```bash
python3 -m unittest discover -s tasks/panoptic/tests -v
```

Run panoptic inference from the `train` directory after downloading the
pretrained semantic RangeNet++ model and SemanticKITTI dataset:

```bash
python3 tasks/panoptic/infer.py \
  --dataset /path/to/SemanticKITTI/dataset \
  --model /path/to/darknet53-1024 \
  --log /path/to/panoptic-predictions \
  --panoptic-config tasks/panoptic/config/semantic-kitti.yaml \
  --instance-method range-image
```
