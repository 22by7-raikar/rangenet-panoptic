import sys
from pathlib import Path

import pytest

PANOPTIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANOPTIC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import write_dataset  # noqa: E402

from lpsad.config import build_label_space, load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(PANOPTIC / "config" / "lpsad-darknet53.yaml")
    cfg["sensor"].update(height=16, width=256)
    cfg["model"]["backbone"]["layers"] = 21
    t = cfg["train"]
    t.update(data_root=str(write_dataset(tmp_path / "kitti")),
             split={"train": [0], "valid": [8], "test": [8]},
             epochs=2, batch_size=2, accumulate=1, workers=0, amp="none",
             val_max_scans=None, log_every=1)
    return cfg


@pytest.fixture
def labels(cfg):
    return build_label_space(cfg)
