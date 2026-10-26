# RangeNet Panoptic

Panoptic segmentation of LiDAR scans on range images. This repository extends
[RangeNet++](https://github.com/PRBonn/lidar-bonnetal) (LiDAR-Bonnetal, Milioto
et al., IROS 2019) from semantic to **panoptic** segmentation with a modern
PyTorch 2 re-implementation of **LPSAD** (Milioto et al., *LiDAR Panoptic
Segmentation for Autonomous Driving*, IROS 2020). LPSAD is the panoptic model the
original authors built on this codebase but never released.

Every point of a scan gets a semantic class (road, car, person, ...) and, for
countable *things* such as cars and people, an instance id.

## Why this project exists

RangeNet++ is a fast and widely used LiDAR semantic segmentation framework. It
projects each scan to a 2D range image and segments it with a 2D CNN, fast
enough for a robot or an embedded GPU. Semantic labels alone don't separate
two parked cars or a group of pedestrians, though, and most driving tasks need
that.

The upstream repository announced panoptic segmentation as "Coming Soon"
and was then archived with the folder still empty. The same authors published
the design they had in mind, LPSAD (IROS 2020). It keeps the RangeNet++ encoder,
adds an instance decoder that regresses 3D offsets to object centres, and
clusters points with learned radii. The paper says the code would be released
in LiDAR-Bonnetal, but it never was, and no third-party implementation exists.

This project started as an extension of RangeNet++ semantic training to
panoptic segmentation, with three requirements:

- **Build on the authors' intended design**, not an ad-hoc head. Their design
  reuses everything RangeNet++ already does well, and their paper gives
  equations, ablations and target numbers to check against.
- **Don't be outdated.** The 2019 stack (PyTorch 1.1, TensorFlow 1.13,
  Python's removed `imp` module) no longer installs. The code targets PyTorch 2
  with mixed precision, and the docs say plainly where range-view methods stand
  against current point-based methods.
- **Train on a laptop.** The target machine has a 6 GB RTX 3060 laptop GPU,
  hence bf16, gradient accumulation and initialising from RangeNet++ weights.

## What is in the repository

| Task | Model | Status |
|---|---|---|
| [Semantic](train/tasks/semantic) | RangeNet++ (SqueezeSeg, SqueezeSegV2, Darknet 21/53), kNN / CRF post-processing | Upstream code, ported to Python 3.12+, NumPy 2 and PyTorch 2. Train, infer and evaluate verified on CPU. |
| [Panoptic](train/tasks/panoptic) | LPSAD: shared Darknet encoder, semantic and instance decoders, trilinear upsampling, learned-radius clustering | New. Tested on synthetic scans; **not yet trained on SemanticKITTI**. |

![System overview](docs/diagrams/system.svg)

## Measured so far

Real KITTI scans (20 from sequence 08), no training. More figures and the
commands to make them are in [docs/figures](docs/figures/README.md).

**The range image keeps only part of each scan.** Each pixel keeps its
closest point; every other point gets its label copied from that pixel.

![Points kept by the range projection](docs/figures/projection.png)

| Width | 64 x 512 | 64 x 1024 | 64 x 2048 | 64 x 4096 |
|---|---|---|---|---|
| Points kept | 21.0% | 41.1% | 77.7% | 90.9% |

**Distance-aware upsampling keeps object borders.** Rebuilding the full-width
range image from half width, LPSAD's upsampling errs 0.30 m at object edges,
against 3.1 m for 2D interpolation, which blends foreground and background.

![Upsampling error at object edges](docs/figures/upsampling.png)

| Upsampling | Mean abs. error, all | at object edges |
|---|---|---|
| nearest | 0.29 m | 3.29 m |
| 2D linear | 0.27 m | 3.11 m |
| LPSAD trilinear | 0.07 m | 0.30 m |

(Range is geometric, so 3D-distance weighting favours it by construction. The
figure shows that features come from the same surface; the paper measures
the effect on panoptic quality, 30.8 → 36.5 PQ.)

## Quick start

```sh
git clone https://github.com/22by7-raikar/rangenet-panoptic && cd rangenet-panoptic
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Get [SemanticKITTI](http://semantic-kitti.org/dataset.html) (the KITTI odometry
velodyne scans plus labels) into `/data/semantic-kitti/sequences/00..21`.

**Panoptic (LPSAD)**, see [its README](train/tasks/panoptic/README.md):

```sh
cd train/tasks/panoptic
python -m pytest tests                      # about 2 min on CPU
python train.py -d /data/semantic-kitti -l logs/lpsad -p /models/darknet53 \
    --set sensor.width=1024 train.batch_size=2 train.accumulate=4 \
          train.freeze_backbone_epochs=10 train.freeze_backbone_bn=true
python infer.py -m logs/lpsad/best.pt -d /data/semantic-kitti -o preds --split valid
python evaluate.py -d /data/semantic-kitti -p preds --split valid
```

**Semantic (RangeNet++)**, see [its README](train/tasks/semantic/README.md):

```sh
cd train/tasks/semantic
python train.py -d /data/semantic-kitti -ac config/arch/darknet53.yaml -l logs/darknet53
python infer.py -d /data/semantic-kitti -l preds -m logs/darknet53
python evaluate_iou.py -d /data/semantic-kitti -p preds --split valid
```

## Status and roadmap

Verified so far, on CPU with synthetic scans:

- The panoptic pipeline runs end to end (train, resume, infer, evaluate), and
  the model overfits a small set: car PQ 0.41 and person PQ 0.80.
- The semantic pipeline runs end to end on PyTorch 2.14 / Python 3.13.
- The panoptic model loads backbone weights written by the semantic trainer.
- Single KITTI sequences, or subsets of one, download without the 85 GB
  archive (tested against the KITTI server).

Next, without training (about 1 GB of data, see
[Real numbers without training](train/tasks/panoptic/README.md#real-numbers-without-training)):

1. Projection and clustering ceilings of the range-image approach on
   sequence 08 (`tools/upper_bounds.py`).
2. A training-free panoptic baseline: pre-trained RangeNet++ semantics plus
   Euclidean clustering (`tools/baseline.py`). This also checks that the 2019
   checkpoints load.
3. Speed and training memory on the 6 GB GPU (`tools/benchmark.py`).

Then:

4. Train LPSAD on a subset of sequences and compare with step 2; reproduce
   the paper's ablations (Tables III, IV) as far as the subset allows.
5. Optional: a query-based mask head, or 4D panoptic (tracking), on the same
   data pipeline.

For context: published LiDAR-only point-based methods reach about 65 PQ on the
SemanticKITTI test set (P3Former, 2023), against 38 PQ for LPSAD. Range-view
models trade accuracy for speed and simple 2D deployment. The goal here is a
faithful, fast, modern baseline, not the state of the art.

## Documentation

| | |
|---|---|
| [Architecture](docs/architecture.md) | System and subsystem design with diagrams |
| [Design decisions](docs/design-decisions.md) | Why things are the way they are, including every deviation from the paper |
| [Code walkthrough](docs/code-walkthrough.md) | One scan through the code: every step, function and tensor shape, paper equations mapped to code |
| [Figures](docs/figures/README.md) | Images and tables for slides, with the commands that make them |
| [Development](docs/development.md) | Setup, tests, docs build, CI, conventions |
| [Panoptic task](train/tasks/panoptic/README.md) | Usage, configuration, reference results |
| [Semantic task](train/tasks/semantic/README.md) | Upstream RangeNet++ usage |
| API reference | `make -C docs html`, then open `docs/_build/html/index.html` (also built by CI). *Topics* groups the API by subsystem, with diagrams |

## Repository layout

```
.
├── README.md                    this file (also the Doxygen main page)
├── Doxyfile                     API reference configuration
├── docs/                        architecture, walkthrough, design decisions, development
│   ├── diagrams/                Graphviz sources (*.dot) and rendered *.svg
│   └── figures/                 result images and tables (from tools/visualize.py)
├── requirements.txt
└── train/
    ├── backbones/               SqueezeSeg, SqueezeSegV2, Darknet encoders (shared)
    ├── common/                  laser scan I/O and projection, logging, utilities
    └── tasks/
        ├── semantic/            RangeNet++ (upstream, ported)
        └── panoptic/            LPSAD
            ├── lpsad/           config, dataset, model, losses, cluster, metrics, trainer
            ├── config/          lpsad-darknet53.yaml
            ├── tools/           data download, training-free baseline, ceilings, benchmark
            ├── tests/           unit + end-to-end tests on synthetic scans
            └── train.py, infer.py, evaluate.py
```

## Pre-trained RangeNet++ models

The upstream semantic models can initialise the panoptic encoder (`-p`). They
are hosted by the University of Bonn:
[darknet21](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet21.tar.gz),
[darknet53](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53.tar.gz),
[darknet53-1024](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-1024.tar.gz),
[darknet53-512](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-512.tar.gz),
[squeezeseg](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/squeezeseg.tar.gz),
[squeezesegV2](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/squeezesegV2.tar.gz)
(add `-crf` before `.tar.gz` for the CRF variants). Predictions for every split
are available under the same path with `predictions/` instead of `models/`.
To enable kNN post-processing for a semantic model, set `post: KNN: use: True`
in its `arch_cfg.yaml`.

## Credits, license and citations

Forked from [PRBonn/lidar-bonnetal](https://github.com/PRBonn/lidar-bonnetal)
by Andres Milioto, Jens Behley, Ignacio Vizzo and Cyrill Stachniss (University
of Bonn), under the MIT license, which this repository keeps (see
[LICENSE](LICENSE)). The panoptic method is theirs too; the implementation here
is independent. Pre-trained models keep the license of the dataset they were
trained on.

If you use this work, please cite the original papers and the dataset:

```bibtex
@inproceedings{milioto2020iros,
  author    = {A. Milioto and J. Behley and C. McCool and C. Stachniss},
  title     = {{LiDAR Panoptic Segmentation for Autonomous Driving}},
  booktitle = {IEEE/RSJ Intl.~Conf.~on Intelligent Robots and Systems (IROS)},
  year      = {2020}
}

@inproceedings{milioto2019iros,
  author    = {A. Milioto and I. Vizzo and J. Behley and C. Stachniss},
  title     = {{RangeNet++: Fast and Accurate LiDAR Semantic Segmentation}},
  booktitle = {IEEE/RSJ Intl.~Conf.~on Intelligent Robots and Systems (IROS)},
  year      = {2019}
}

@inproceedings{behley2019iccv,
  author    = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
  title     = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
  booktitle = {Proc. of the IEEE/CVF International Conf.~on Computer Vision (ICCV)},
  year      = {2019}
}
```
