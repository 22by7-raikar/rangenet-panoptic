"""Config loading and the label look-up tables derived from it."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class LabelSpace:
    """Everything needed to go between raw SemanticKITTI labels and classes."""

    names: list[str]            # xentropy class names, index = class id
    learning_lut: np.ndarray    # raw semantic id -> xentropy class
    inverse_lut: np.ndarray     # xentropy class -> raw semantic id
    ignore: list[int]           # xentropy classes ignored in losses/metrics
    content: np.ndarray         # point frequency per xentropy class
    things: list[int]
    class_to_category: np.ndarray  # xentropy class -> category (0 = ignore)
    category_names: list[str]

    @property
    def nclasses(self) -> int:
        return len(self.names)

    @property
    def ncategories(self) -> int:
        return len(self.category_names)

    def is_thing(self) -> np.ndarray:
        mask = np.zeros(self.nclasses, dtype=bool)
        mask[self.things] = True
        return mask


def _lut(mapping: dict, size: int | None = None) -> np.ndarray:
    size = size or (1 << 16)  # semantic ids are the lower 16 bits
    lut = np.zeros(size, dtype=np.int64)
    for k, v in mapping.items():
        lut[k] = v
    return lut


def load_config(path: str | Path) -> dict:
    """Read a task config and the label config it references (``labels``)."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    labels = Path(cfg["labels"])
    if not labels.is_absolute():
        labels = (path.parent / labels).resolve()
    with open(labels) as f:
        cfg["label_cfg"] = yaml.safe_load(f)
    cfg["config_path"] = str(path)
    return cfg


def build_label_space(cfg: dict) -> LabelSpace:
    """Derive class maps, frequencies, things and categories from a config."""
    lab = cfg["label_cfg"]
    inv = lab["learning_map_inv"]
    nclasses = len(inv)
    names = [lab["labels"][inv[c]] for c in range(nclasses)]

    content = np.zeros(nclasses, dtype=np.float64)
    for raw, freq in lab["content"].items():
        content[lab["learning_map"][raw]] += freq

    cats = cfg["panoptic"]["categories"]
    class_to_category = np.zeros(nclasses, dtype=np.int64)
    category_names = ["unlabeled"] + [""] * len(cats)
    for cid, cat in cats.items():
        category_names[int(cid)] = cat["name"]
        for c in cat["classes"]:
            class_to_category[c] = int(cid)

    return LabelSpace(
        names=names,
        learning_lut=_lut(lab["learning_map"]),
        inverse_lut=_lut(inv, nclasses),
        ignore=[c for c, ign in lab["learning_ignore"].items() if ign],
        content=content,
        things=list(cfg["panoptic"]["things"]),
        class_to_category=class_to_category,
        category_names=category_names,
    )
