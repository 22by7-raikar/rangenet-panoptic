"""LPSAD: single-stage LiDAR panoptic segmentation on range images.

Re-implementation of Milioto, Behley, McCool and Stachniss, "LiDAR Panoptic
Segmentation for Autonomous Driving", IROS 2020, on top of the RangeNet++
(LiDAR-Bonnetal) Darknet backbones, using a modern PyTorch 2.x training stack.
"""

import sys
from pathlib import Path

# make the shared LiDAR-Bonnetal modules (train/backbones, train/common)
# importable without the old "import __init__ as booger" trick.
TRAIN_ROOT = Path(__file__).resolve().parents[3]
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))
