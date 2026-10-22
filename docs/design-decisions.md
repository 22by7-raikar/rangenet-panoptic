# Design decisions

Each record states the context, the decision and its consequences, so later
changes can revisit a decision knowing why it was made. Records about the
paper say whether they **follow** it or **deviate** from it.

| # | Decision | Kind |
|---|---|---|
| 1 | Rebuild LPSAD rather than design a new panoptic head | scope |
| 2 | Panoptic as a self-contained task package | structure |
| 3 | Keep the upstream layout and code style of the semantic task | structure |
| 4 | Do not detach skip connections | deviation from upstream |
| 5 | Trilinear upsampling neighbourhood | interpretation |
| 6 | Clustering radius: seed + candidate | deviation |
| 7 | Bounded repel loss | deviation |
| 8 | Class weights 1 / ln(1.02 + f) | interpretation |
| 9 | Instance centres from the full point cloud | interpretation |
| 10 | Radius scale estimated during training | follows paper |
| 11 | Training for a 6 GB GPU | engineering |
| 12 | Own PQ evaluator, official script for reported numbers | engineering |
| 13 | Diagrams as Graphviz sources, API docs with Doxygen | engineering |

## 1. Rebuild LPSAD rather than design a new panoptic head

**Context.** The goal was to extend RangeNet++ to panoptic segmentation without
producing outdated work. The options were a new head (for example a
query-based mask head, as in MaskRange or MaskPLS) or the authors' own LPSAD
design, which was published but never released.

**Decision.** Re-implement LPSAD first.

**Consequences.** The paper gives equations, ablations (Tables III, IV) and
target numbers (38.0 PQ test, 36.5 PQ validation) to check the implementation
against. It reuses the RangeNet++ encoder and pre-trained weights. The data
pipeline, evaluation and training loop are independent of the head, so a mask
head can be added later as an alternative and compared against a known
baseline. LPSAD is not state of the art (point-based methods reach about
65 PQ); the docs say so.

## 2. Panoptic as a self-contained task package

**Context.** Upstream tasks load modules by file path at runtime (`imp`) and
import `__init__.py` to patch `sys.path`.

**Decision.** `train/tasks/panoptic/lpsad` is a normal Python package with
explicit imports. Its only dependency on the rest of the repository is
`backbones/darknet.py`.

**Consequences.** The package can be tested with `pytest` from a clean
checkout, its modules have clear dependencies (see
[architecture](architecture.md#panoptic-package-structure)), and changes to the
semantic task can't break it.

## 3. Keep the upstream layout and code style of the semantic task

**Context.** The semantic task is upstream code (2-space indentation, runtime
module loading). It didn't run on current Python, NumPy or PyTorch.

**Decision.** Port it with the smallest possible changes:
- `imp.load_source` becomes an `importlib`-based `common/module_loader.py`;
- `np.float` becomes `np.float64`;
- the TensorFlow 1 logger becomes `torch.utils.tensorboard` behind the same
  interface;
- labels are cast to `long` on CPU too.

No restructuring, no reformatting.

**Consequences.** Diffs against upstream stay small and reviewable, and
upstream knowledge (issues, forks) still applies. The semantic task keeps its
older style; the new code uses PEP 8.

## 4. Do not detach skip connections

**Context.** Upstream `Backbone.run_layer` stores `x.detach()` as the skip,
so the encoder only learns through its bottleneck. The paper trains the shared
encoder with all losses.

**Decision.** `DarknetBackbone` overrides `run_layer` without `detach()`.
Parameter names are unchanged.

**Consequences.** Instance and semantic losses reach the early layers directly.
RangeNet++ checkpoints still load strictly. The semantic task keeps upstream
behaviour.

## 5. Trilinear upsampling neighbourhood

**Context.** The paper (Fig. 4) blends "the four closest feature values in the
lower resolution grid" by inverse 3D distance. The encoder downsamples only
horizontally, so a fine pixel lies between two coarse columns in the same row.

**Decision.** Candidates are the two bracketing coarse columns in rows
`h-1, h, h+1`, wrapping around at 360°. Keep the 4 closest in 3D and weight
them by `1 / (d + 1e-3)`, normalised. Empty fine pixels, or pixels without a
valid candidate, use the nearest coarse pixel. Weights are computed without
gradient. Configurable as `trilinear_rows` and `trilinear_k`.

**Consequences.** It matches the paper's "4 closest" while respecting the
horizontal-only downsampling. Memory stays linear in the feature map, because
the weights need no gradient.

## 6. Clustering radius: seed + candidate

**Context.** In the paper a pixel joins the seed's instance if its predicted
centre lies within the *seed's* learned radius, and the radius learns the
seed's own offset error. Two predictions with errors `e1` and `e2` from the
true centre can be up to `e1 + e2` apart. In tests the paper's rule split
objects into many fragments: car PQ 0.00 on an overfitted synthetic model.

**Decision.** Default `cluster.radius_mode: sum`: compare against seed radius
plus candidate radius. `seed` reproduces the paper.

**Consequences.** On the same model car PQ rose to 0.38 (both features) and
0.48 (centres only). This is a deliberate deviation. Compare both modes on
SemanticKITTI with `infer.py --cluster radius_mode=seed`.

## 7. Bounded repel loss

**Context.** Eq. 6 is `1 / ||e_A - e_B||^2`, which is unbounded when two
instance embeddings coincide, as they do at initialisation.

**Decision.** Use `1 / (||e_A - e_B||^2 + delta)` with `loss.repel_delta: 1.0`,
over pairs of instances in the same scan.

**Consequences.** Stable gradients from the first step. Setting `delta` close
to 0 recovers the paper.

## 8. Class weights 1 / ln(1.02 + f)

**Context.** Eq. 4 writes `w_c = 1 / log(f_c + eps)`, which is negative for
`f_c + eps < 1`. The intent, "inverse class frequency, bounded", matches the
ENet weighting `1 / ln(c + f)` with `c = 1.02`.

**Decision.** ENet form with `loss.class_weight_c: 1.02`. Categories use the
same form. Ignored classes get weight 0, and the loss is normalised by the
weights of the targets.

## 9. Instance centres from the full point cloud

**Context.** The paper doesn't say whether the centre `c_i` is computed from
all points of an instance or only from the ones visible in the range image.

**Decision.** All points, computed before projection, after augmentation.

**Consequences.** Targets don't depend on the image resolution or on which
points win a pixel.

## 10. Radius scale estimated during training

**Follows the paper.** Learned radii are scaled by the ratio of validation to
training error. `Trainer.validate` computes it each epoch, stores it in the
checkpoint, and `infer.py` uses it. To avoid storing all validation outputs,
validation clusters with the previous epoch's scale. `radius_scale` can also
be set to a number.

## 11. Training for a 6 GB GPU

**Context.** The target GPU is an RTX 3060 laptop GPU (6 GB). RangeNet++ trains
Darknet53 at 64 x 2048 with batch 8.

**Decision.**
- bf16 autocast (Ampere supports it, and it needs no loss scaling);
- gradient accumulation;
- backbone initialised from RangeNet++, with optional freezing and frozen BN
  statistics for small batches;
- 64 x 1024 recommended for experiments, 64 x 512 for debugging only.

**Consequences.** RangeNet++ Table I shows the reason for the resolution
choice: person IoU drops from 38.3 (2048) to 29.6 (1024) to 18.1 (512), and
small things drive panoptic quality. Memory use on the real GPU is still
unverified.

## 12. Own PQ evaluator, official script for reported numbers

**Context.** Validation during training needs PQ without shelling out to
another repository.

**Decision.** `lpsad/metrics.py` follows semantic-kitti-api's `PanopticEval`
logic: drop ignored ground truth, shift instance ids by one, match at
IoU > 0.5, `min_points` 50. Unit tests cover the hand-computed cases.

**Consequences.** Fast in-loop validation. For published numbers, run the
official `evaluate_panoptic.py`.

## 13. Diagrams as Graphviz sources, API docs with Doxygen

**Decision.** Diagrams live as `docs/diagrams/*.dot` with the rendered SVGs
committed. GitHub shows the SVGs in Markdown, and Doxygen uses the same files.
`make -C docs diagrams` re-renders them after a `.dot` file changes. CI renders
them fresh for the API docs and fails if a source doesn't render. It doesn't
diff the committed SVGs, because Graphviz versions differ in their output.
Doxygen builds an API reference from the Python docstrings and includes these
Markdown pages.

**Consequences.** One source per diagram, diffable, no binary images, and no
diagrams drifting from the code they describe.
