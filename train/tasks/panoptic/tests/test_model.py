import copy

import torch

from lpsad.dataset import SemanticKittiPanoptic, collate
from lpsad.model import LPSAD, TrilinearUpsample, point_pyramid


def _geometry(B=1, H=4, Wc=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    xyz_f = torch.randn(B, 3, H, 2 * Wc, generator=g) * 5
    mask_f = torch.ones(B, H, 2 * Wc, dtype=torch.bool)
    xyz_c, mask_c = xyz_f[..., ::2], mask_f[..., ::2]
    return xyz_c, mask_c, xyz_f, mask_f


def test_point_pyramid():
    xyz = torch.randn(2, 3, 4, 64)
    mask = torch.ones(2, 4, 64, dtype=torch.bool)
    pyr = point_pyramid(xyz, mask, [2, 2, 1, 2, 2])
    assert sorted(pyr) == [1, 2, 4, 8, 16]
    assert pyr[16][0].shape == (2, 3, 4, 4)
    assert torch.equal(pyr[2][0], xyz[..., ::2])


def test_trilinear_weights_normalised_and_exact_points_win():
    up = TrilinearUpsample(rows=3, k=4)
    xyz_c, mask_c, xyz_f, mask_f = _geometry()
    w = up.weights(xyz_c, mask_c, xyz_f, mask_f)
    assert w.shape == (1, 6, 4, 16)
    torch.testing.assert_close(w.sum(1), torch.ones(1, 4, 16))
    assert ((w > 0).sum(1) <= 4).all()
    # even fine columns coincide with a coarse point: (row 0, column j0)
    assert (w[:, 2, :, ::2] > 0.99).all()


def test_trilinear_constant_field_and_gradient():
    up = TrilinearUpsample()
    xyz_c, mask_c, xyz_f, mask_f = _geometry(B=2)
    x = torch.full((2, 5, 4, 8), 3.0, requires_grad=True)
    y = up(x, xyz_c, mask_c, xyz_f, mask_f)
    assert y.shape == (2, 5, 4, 16)
    torch.testing.assert_close(y, torch.full_like(y, 3.0))
    y.sum().backward()
    torch.testing.assert_close(x.grad.sum(), torch.tensor(2 * 5 * 4 * 16.0))


def test_trilinear_fallbacks_and_wraparound():
    up = TrilinearUpsample(rows=1, k=2)
    xyz_c, mask_c, xyz_f, mask_f = _geometry(H=1, Wc=4)
    # last fine column sits between coarse column 3 and (wrapped) column 0
    xyz_f[..., -1] = xyz_c[..., 0]
    w = up.weights(xyz_c, mask_c, xyz_f, mask_f)
    assert w[0, 1, 0, -1] > 0.99
    # empty fine pixel -> nearest (column j0)
    mask_f[..., 3] = False
    w = up.weights(xyz_c, mask_c, xyz_f, mask_f)
    assert w[0, 0, 0, 3] == 1 and w[0, 1, 0, 3] == 0
    # no valid candidate -> nearest
    mask_c[..., :] = False
    w = up.weights(xyz_c, mask_c, xyz_f, mask_f)
    assert (w[:, 0] == 1).all()


def _batch(cfg, labels):
    ds = SemanticKittiPanoptic(cfg["train"]["data_root"], [0], cfg, labels)
    return collate([ds[0], ds[1]])


def test_forward_backward(cfg, labels):
    model = LPSAD(cfg, labels.nclasses, labels.ncategories)
    batch = _batch(cfg, labels)
    out = model(batch["input"], batch["xyz"], batch["mask"])
    H, W = cfg["sensor"]["height"], cfg["sensor"]["width"]
    assert out["logits"].shape == (2, labels.nclasses, H, W)
    assert out["cat_logits"].shape == (2, labels.ncategories, H, W)
    assert out["embedding"].shape == (2, 32, H, W)
    assert out["offset"].shape == (2, 3, H, W)
    assert (out["offset_err"] >= 0).all() and (out["embedding_err"] >= 0).all()
    sum(v.sum() for v in out.values()).backward()
    # skips are not detached: the first encoder stage receives gradient
    # through the OS-2 skip as well as through the bottleneck
    assert model.backbone.enc1.conv.weight.grad.abs().sum() > 0


def test_transpose_and_lower_output_stride(cfg, labels):
    cfg = copy.deepcopy(cfg)
    cfg["model"]["decoder"]["upsample"] = "transpose"
    cfg["model"]["backbone"]["OS"] = 8
    model = LPSAD(cfg, labels.nclasses, labels.ncategories)
    batch = _batch(cfg, labels)
    out = model(batch["input"], batch["xyz"], batch["mask"])
    assert out["logits"].shape[-2:] == batch["mask"].shape[-2:]


def test_rangenet_backbone_checkpoint_loads(cfg, labels, tmp_path):
    # a RangeNet++ model dir stores the bare backbone state dict as "backbone"
    model = LPSAD(cfg, labels.nclasses, labels.ncategories)
    torch.save(model.backbone.state_dict(), tmp_path / "backbone")
    other = LPSAD(cfg, labels.nclasses, labels.ncategories)
    other.load_rangenet_backbone(tmp_path)
    for a, b in zip(model.backbone.parameters(), other.backbone.parameters()):
        assert torch.equal(a, b)
