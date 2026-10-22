import numpy as np
import torch

from lpsad.cluster import extract_instances
from lpsad.losses import LPSADLoss
from lpsad.metrics import PanopticEval


def _toy(B=1, H=2, W=8, E=4):
    """Two car instances (ids 1, 2) on a row of road pixels."""
    sem = torch.full((B, H, W), 9)
    inst = torch.zeros(B, H, W, dtype=torch.long)
    sem[:, 0, :3], inst[:, 0, :3] = 1, 1
    sem[:, 0, 5:], inst[:, 0, 5:] = 1, 2
    offset = torch.randn(B, 3, H, W) * (inst > 0).unsqueeze(1)
    batch = {"mask": torch.ones(B, H, W, dtype=torch.bool), "sem": sem,
             "cat": torch.where(sem == 1, 1, 3), "inst": inst, "offset": offset}
    out = {
        "logits": torch.randn(B, 20, H, W), "cat_logits": torch.randn(B, 7, H, W),
        "embedding": torch.zeros(B, E, H, W), "embedding_err": torch.zeros(B, 1, H, W),
        "offset": offset.clone(), "offset_err": torch.zeros(B, 1, H, W),
    }
    return out, batch


def test_perfect_offsets_give_zero_center_losses(cfg, labels):
    loss = LPSADLoss(cfg, labels)
    out, batch = _toy()
    total, terms, stats = loss(out, batch)
    assert torch.isfinite(total)
    assert terms["center"] == 0 and terms["center_err"] == 0
    assert terms["attract"] == 0 and terms["embed_err"] == 0
    assert stats["center_err_mean"] == 0


def test_repel_falls_as_instances_separate(cfg, labels):
    loss = LPSADLoss(cfg, labels)
    out, batch = _toy()
    values = []
    for d in (0.0, 1.0, 5.0):
        out["embedding"] = torch.zeros_like(out["embedding"])
        out["embedding"][:, 0][batch["inst"] == 2] = d
        values.append(loss(out, batch)[1]["repel"].item())
    assert values[0] > values[1] > values[2]


def test_repel_only_within_a_scan(cfg, labels):
    loss = LPSADLoss(cfg, labels)
    out, batch = _toy(B=2)
    batch["inst"][1] = 0  # second scan has no instances
    batch["inst"][0, 0, 5:] = 0  # first scan has one
    _, terms, _ = loss(out, batch)
    assert terms["repel"] == 0


def test_no_instances_is_finite(cfg, labels):
    loss = LPSADLoss(cfg, labels)
    out, batch = _toy()
    batch["inst"].zero_()
    total, terms, stats = loss(out, batch)
    assert torch.isfinite(total) and terms["center"] == 0 and stats == {}


def _clusters(features, learned=True):
    H, W = 1, 40
    sem = torch.full((H, W), 9)
    sem[0, :10] = 1
    sem[0, 10:20] = 1
    sem[0, 20:30] = 6  # a person right next to the cars
    xyz = torch.zeros(3, H, W)
    xyz[0] = torch.linspace(0, 20, W)
    centers = torch.zeros(3, H, W)
    centers[0, 0, :10], centers[0, 0, 10:20], centers[0, 0, 20:30] = 2, 7, 7
    emb = torch.zeros(4, H, W)
    emb[0, 0, 10:20] = 3.0
    return extract_instances(
        sem, torch.ones(H, W, dtype=torch.bool), torch.arange(20).lt(9) & torch.arange(20).gt(0),
        xyz, xyz - centers, torch.full((1, H, W), 0.5), emb, torch.full((1, H, W), 0.5),
        features=features, learned_radius=learned,
        generator=torch.Generator().manual_seed(0))


def test_extract_instances_separates_objects_and_classes():
    for features in ("center", "both"):
        inst = _clusters(features)[0]
        assert (inst[30:] == 0).all()           # stuff
        a, b, c = inst[:10], inst[10:20], inst[20:30]
        for seg in (a, b, c):
            assert (seg == seg[0]).all() and seg[0] > 0
        # same centre as the second car, but a different class
        assert len({a[0].item(), b[0].item(), c[0].item()}) == 3


def test_extract_instances_embedding_only_and_fixed_radius():
    inst = _clusters("embedding", learned=False)[0]
    assert inst[0] != inst[10] and (inst[:10] == inst[0]).all()


def test_radius_mode_sum_uses_both_radii():
    # one car; centre predictions alternate +/-0.4 m around the true centre,
    # each with a learned radius of 0.5: pairs are 0.8 m apart, which is
    # outside one radius but inside the sum of both.
    H, W = 1, 10
    sem = torch.ones(H, W, dtype=torch.long)
    xyz = torch.zeros(3, H, W)
    centers = torch.zeros(3, H, W)
    centers[0, 0, ::2], centers[0, 0, 1::2] = 0.4, -0.4
    is_thing = torch.arange(20).lt(9) & torch.arange(20).gt(0)
    args = (sem, torch.ones(H, W, dtype=torch.bool), is_thing,
            xyz, xyz - centers, torch.full((1, H, W), 0.5), torch.zeros(4, H, W),
            torch.full((1, H, W), 0.5))
    seed = extract_instances(*args, features="center", radius_mode="seed",
                             generator=torch.Generator().manual_seed(0))
    total = extract_instances(*args, features="center", radius_mode="sum",
                              generator=torch.Generator().manual_seed(0))
    assert len(seed.unique()) == 2   # split into the two halves
    assert len(total.unique()) == 1 and total.min() > 0


def test_panoptic_eval_perfect_and_split():
    ev = PanopticEval(4, ignore=[0], things=[1], min_points=30)
    sem = np.array([1] * 100 + [2] * 50 + [0] * 10)
    inst = np.array([5] * 100 + [0] * 60)
    ev.add(sem, inst, sem, inst)
    r = ev.results()
    assert r["PQ"] == 1.0 / 1.0 * (2 / 3)  # class 3 is absent: PQ 0
    assert r["PQ_th"] == 1.0 and r["mIoU"] == 2 / 3

    ev = PanopticEval(3, ignore=[0], things=[1], min_points=30)
    pred_inst = np.array([1] * 60 + [2] * 40 + [0] * 60)
    ev.add(sem[:160], pred_inst, sem[:160], inst)
    car = ev.results()["per_class"]
    # IoU 0.6 match (TP) + 40-point fragment (FP): PQ = 0.6 / 1.5
    assert np.isclose(car["PQ"][1], 0.4)
    assert np.isclose(car["SQ"][1], 0.6)


def test_panoptic_eval_ignores_unlabeled_ground_truth():
    ev = PanopticEval(3, ignore=[0], things=[1], min_points=1)
    gt_sem = np.array([1, 1, 0, 0])
    pred_sem = np.array([1, 1, 1, 1])
    ev.add(pred_sem, np.array([1, 1, 1, 1]), gt_sem, np.array([1, 1, 0, 0]))
    assert ev.results()["per_class"]["PQ"][1] == 1.0
