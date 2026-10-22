import numpy as np
import torch

from lpsad.dataset import SemanticKittiPanoptic, collate, instance_targets, spherical_projection


def test_projection_keeps_closest_point():
    # two points along the same ray: the closer one must own the pixel
    xyz = np.array([[10.0, 0.0, -1.0], [5.0, 0.0, -0.5]], dtype=np.float32)
    px, py, depth, idx = spherical_projection(xyz, 64, 1024, 3, -25)
    assert (px[0], py[0]) == (px[1], py[1])
    assert idx[py[0], px[0]] == 1
    assert (idx >= 0).sum() == 1


def test_instance_targets_point_to_full_cloud_centre():
    xyz = np.array([[0, 0, 0], [2, 0, 0], [10, 10, 0], [5, 5, 5]], dtype=np.float32)
    raw = np.array([10 | (1 << 16), 10 | (1 << 16), 10 | (2 << 16), 40], dtype=np.int64)
    sem = np.array([1, 1, 1, 9])
    is_thing = np.zeros(20, dtype=bool)
    is_thing[1:9] = True
    inst, off = instance_targets(xyz, raw, sem, is_thing)
    assert inst.tolist()[:3] == [1, 1, 2] and inst[3] == 0
    np.testing.assert_allclose(off[0], [-1, 0, 0])
    np.testing.assert_allclose(off[1], [1, 0, 0])
    np.testing.assert_allclose(off[2:], 0)


def test_dataset_item(cfg, labels):
    ds = SemanticKittiPanoptic(cfg["train"]["data_root"], [0], cfg, labels)
    item = ds[0]
    H, W = cfg["sensor"]["height"], cfg["sensor"]["width"]
    assert item["input"].shape == (5, H, W)
    assert item["offset"].shape == (3, H, W)
    assert item["mask"].dtype == torch.bool

    # every pixel of an instance points to the same centre
    inst, xyz, off = item["inst"], item["xyz"], item["offset"]
    for i in inst.unique()[1:]:
        centers = (xyz - off)[:, inst == i]
        assert torch.allclose(centers, centers[:, :1].expand_as(centers), atol=1e-4)
    # instances only on thing classes; categories follow the config
    assert torch.isin(item["sem"][inst > 0], torch.tensor(labels.things)).all()
    car = item["sem"] == 1
    assert car.any() and (item["cat"][car] == 1).all()
    # empty pixels carry nothing
    assert (item["input"][:, ~item["mask"]] == 0).all()
    assert len(item["point_sem"]) == len(item["px"])

    batch = collate([ds[0], ds[1]])
    assert batch["input"].shape == (2, 5, H, W)
    assert isinstance(batch["px"], list)
