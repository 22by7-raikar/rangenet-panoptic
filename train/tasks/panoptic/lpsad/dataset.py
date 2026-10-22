"""SemanticKITTI panoptic dataset: range-image projection plus instance targets."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import LabelSpace


def spherical_projection(xyz: np.ndarray, H: int, W: int,
                         fov_up_deg: float, fov_down_deg: float):
    """Project points to an H x W range image (paper Eq. 1).

    Points are written in decreasing range so that the closest point of each
    pixel wins. Returns per-point pixel coordinates and, per pixel, the index
    of the point stored there (-1 if empty).
    """
    fov_up = fov_up_deg / 180.0 * np.pi
    fov_down = fov_down_deg / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    depth = np.linalg.norm(xyz, axis=1)
    yaw = -np.arctan2(xyz[:, 1], xyz[:, 0])
    pitch = np.arcsin(np.clip(xyz[:, 2] / np.maximum(depth, 1e-8), -1.0, 1.0))

    px = 0.5 * (yaw / np.pi + 1.0) * W
    py = (1.0 - (pitch + abs(fov_down)) / fov) * H
    px = np.clip(np.floor(px), 0, W - 1).astype(np.int64)
    py = np.clip(np.floor(py), 0, H - 1).astype(np.int64)

    order = np.argsort(depth)[::-1]
    proj_idx = np.full((H, W), -1, dtype=np.int64)
    proj_idx[py[order], px[order]] = order
    return px, py, depth, proj_idx


class Augment:
    """Global point cloud augmentation applied before projection."""

    def __init__(self, rotate=True, flip=True, scale=(0.95, 1.05), rng=None):
        self.rotate = rotate
        self.flip = flip
        self.scale = scale
        self.rng = rng or np.random.default_rng()

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        xyz = xyz.copy()
        if self.rotate:
            a = self.rng.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(a), np.sin(a)
            xyz[:, :2] = xyz[:, :2] @ np.array([[c, s], [-s, c]], dtype=xyz.dtype)
        if self.flip and self.rng.random() < 0.5:
            xyz[:, 1] = -xyz[:, 1]
        if self.scale:
            xyz *= self.rng.uniform(*self.scale)
        return xyz


def instance_targets(xyz: np.ndarray, raw_label: np.ndarray, sem: np.ndarray,
                     is_thing: np.ndarray):
    """Compact per-scan instance ids and offsets to the instance centre.

    Instances are thing points with a non-zero instance id; the centre is the
    mean of *all* points of the instance in the full cloud, not just the ones
    that survive the projection. Returns inst (0 = no instance) and the
    offset x_p - c_i (zero for non-instance points).
    """
    inst = np.zeros(len(xyz), dtype=np.int64)
    offset = np.zeros_like(xyz)
    thing = is_thing[sem] & ((raw_label >> 16) > 0)
    if not thing.any():
        return inst, offset
    _, compact = np.unique(raw_label[thing], return_inverse=True)
    inst[thing] = compact + 1
    k = inst.max() + 1
    sums = np.zeros((k, 3), dtype=np.float64)
    np.add.at(sums, inst[thing], xyz[thing])
    counts = np.bincount(inst[thing], minlength=k)
    centers = (sums / np.maximum(counts, 1)[:, None]).astype(xyz.dtype)
    offset[thing] = xyz[thing] - centers[inst[thing]]
    return inst, offset


class SemanticKittiPanoptic(Dataset):
    def __init__(self, root: str | Path, sequences: list[int], cfg: dict,
                 labels: LabelSpace, gt: bool = True,
                 augment: Augment | None = None, max_scans: int | None = None):
        self.labels = labels
        self.gt = gt
        self.augment = augment
        sensor = cfg["sensor"]
        self.H, self.W = sensor["height"], sensor["width"]
        self.fov_up, self.fov_down = sensor["fov_up"], sensor["fov_down"]
        self.means = np.asarray(sensor["img_means"], dtype=np.float32)
        self.stds = np.asarray(sensor["img_stds"], dtype=np.float32)
        self.is_thing = labels.is_thing()

        root = Path(root).expanduser() / "sequences"
        if not root.is_dir():
            raise FileNotFoundError(f"no sequences folder at {root}")
        self.scans: list[Path] = []
        for seq in sequences:
            seq_dir = root / f"{int(seq):02d}"
            scans = sorted((seq_dir / "velodyne").glob("*.bin"))
            if gt:
                missing = [s for s in scans if not self._label_path(s).exists()]
                if missing:
                    raise FileNotFoundError(f"{len(missing)} scans in {seq_dir} have no label")
            self.scans.extend(scans)
        if max_scans is not None and len(self.scans) > max_scans:
            # evenly spaced subset, deterministic
            idx = np.linspace(0, len(self.scans) - 1, max_scans).round().astype(int)
            self.scans = [self.scans[i] for i in idx]

    @staticmethod
    def _label_path(scan: Path) -> Path:
        return scan.parent.parent / "labels" / (scan.stem + ".label")

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index: int) -> dict:
        path = self.scans[index]
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        xyz, remission = scan[:, :3], scan[:, 3]
        if self.augment is not None:
            xyz = self.augment(xyz)

        px, py, depth, proj_idx = spherical_projection(
            xyz, self.H, self.W, self.fov_up, self.fov_down)
        mask = proj_idx >= 0
        idx = proj_idx[mask]

        image = np.zeros((5, self.H, self.W), dtype=np.float32)
        image[0][mask] = depth[idx]
        image[1:4][:, mask] = xyz[idx].T
        image[4][mask] = remission[idx]
        proj_xyz = image[1:4].copy()
        image = (image - self.means[:, None, None]) / self.stds[:, None, None]
        image *= mask[None]

        item = {
            "input": torch.from_numpy(image),
            "xyz": torch.from_numpy(proj_xyz),
            "mask": torch.from_numpy(mask),
            "px": torch.from_numpy(px),
            "py": torch.from_numpy(py),
            "seq": path.parent.parent.name,
            "name": path.stem,
        }
        if not self.gt:
            return item

        raw = np.fromfile(self._label_path(path), dtype=np.uint32).astype(np.int64)
        if len(raw) != len(xyz):
            raise ValueError(f"{path}: {len(xyz)} points but {len(raw)} labels")
        sem = self.labels.learning_lut[raw & 0xFFFF]
        inst, offset = instance_targets(xyz, raw, sem, self.is_thing)

        def to_image(values, fill=0):
            out = np.full((self.H, self.W) + values.shape[1:], fill, dtype=values.dtype)
            out[mask] = values[idx]
            return out

        proj_sem = to_image(sem)
        item.update({
            "sem": torch.from_numpy(proj_sem),
            "cat": torch.from_numpy(self.labels.class_to_category[proj_sem]),
            "inst": torch.from_numpy(to_image(inst)),
            "offset": torch.from_numpy(np.ascontiguousarray(to_image(offset).transpose(2, 0, 1))),
            "point_sem": torch.from_numpy(sem),
            "point_inst": torch.from_numpy(inst),
        })
        return item


IMAGE_KEYS = ("input", "xyz", "mask", "sem", "cat", "inst", "offset")


def collate(batch: list[dict]) -> dict:
    """Stack range images; keep variable-length per-point data as lists."""
    out = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        out[key] = torch.stack(values) if key in IMAGE_KEYS else values
    return out
