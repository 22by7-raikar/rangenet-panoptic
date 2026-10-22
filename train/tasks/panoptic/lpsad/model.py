"""LPSAD network: shared Darknet encoder, semantic and instance decoders.

Sec. III-B/C of the paper. Both decoders upsample with the distance-aware
trilinear upsampling, which interpolates coarse features with weights from
real 3D distances taken from a point cloud pyramid built alongside the
backbone.
"""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import TRAIN_ROOT  # noqa: F401  (sets up sys.path for backbones/)
from backbones.darknet import Backbone as _DarknetBackbone, BasicBlock


class DarknetBackbone(_DarknetBackbone):
    """RangeNet++ Darknet encoder that keeps gradients through skips.

    The original stores ``x.detach()`` as skip connection, so the encoder only
    learns through the bottleneck. The paper trains the shared backbone with
    all losses, so skips stay attached here. Parameter names are unchanged,
    so RangeNet++ checkpoints load as-is.
    """

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x
            os *= 2
        return y, skips, os


def point_pyramid(xyz: torch.Tensor, mask: torch.Tensor, strides: list[int]):
    """Point cloud pyramid (Sec. III-B).

    A 3x3 convolution with horizontal stride 2 and padding 1 centres output
    column j on input column 2j, so the point at each kernel centre is
    ``xyz[..., ::2]``. Returns {output stride: (xyz, mask)}.
    """
    pyramid = {1: (xyz, mask)}
    os = 1
    for s in strides:
        if s == 2:
            xyz, mask = xyz[..., ::2], mask[..., ::2]
            os *= 2
            pyramid[os] = (xyz, mask)
    return pyramid


class TrilinearUpsample(nn.Module):
    """Distance-aware "trilinear" x2 horizontal upsampling (Sec. III-C, Fig. 4).

    For each fine pixel, candidate coarse pixels are the two coarse columns
    bracketing it in ``rows`` neighbouring rows. The ``k`` candidates closest
    in 3D are combined with weights proportional to the inverse 3D distance.
    The range image wraps around horizontally (360 degrees), so the right-most
    column borrows from the left-most one. Empty fine pixels, or fine pixels
    without any valid candidate, fall back to nearest-neighbour upsampling.

    The weights depend only on the input geometry, so there is nothing to
    learn and no gradient through them.
    """

    def __init__(self, rows: int = 3, k: int = 4, eps: float = 1e-3):
        super().__init__()
        assert rows % 2 == 1, "rows must be odd"
        self.row_offsets = list(range(-(rows // 2), rows // 2 + 1))
        self.k = k
        self.eps = eps

    @staticmethod
    def _shift_rows(t: torch.Tensor, dr: int) -> torch.Tensor:
        """out[..., h, :] = t[..., h + dr, :], zero outside the image."""
        if dr == 0:
            return t
        out = torch.zeros_like(t)
        if dr > 0:
            out[..., :-dr, :] = t[..., dr:, :]
        else:
            out[..., -dr:, :] = t[..., :dr, :]
        return out

    def _candidates(self, Wc: int, device):
        w = torch.arange(2 * Wc, device=device)
        j0 = torch.div(w, 2, rounding_mode="floor")
        j1 = (j0 + 1) % Wc
        return [(dr, j) for dr in self.row_offsets for j in (j0, j1)]

    @torch.no_grad()
    def weights(self, xyz_c, mask_c, xyz_f, mask_f) -> torch.Tensor:
        """(B, n_candidates, H, Wf) interpolation weights, summing to 1."""
        cands = self._candidates(xyz_c.shape[-1], xyz_c.device)
        dists = []
        for dr, j in cands:
            cx = self._shift_rows(xyz_c, dr).index_select(-1, j)
            cm = self._shift_rows(mask_c, dr).index_select(-1, j)
            d = torch.linalg.vector_norm(xyz_f.float() - cx.float(), dim=1)
            dists.append(torch.where(cm, d, torch.full_like(d, float("inf"))))
        dist = torch.stack(dists, dim=1)

        if self.k < dist.shape[1]:
            kth = dist.topk(self.k, dim=1, largest=False).values[:, -1:]
            dist = torch.where(dist <= kth, dist, torch.full_like(dist, float("inf")))
        w = torch.where(torch.isfinite(dist), 1.0 / (dist + self.eps), torch.zeros_like(dist))
        total = w.sum(dim=1, keepdim=True)

        nearest = torch.zeros_like(w)
        nearest[:, 2 * self.row_offsets.index(0)] = 1.0  # same row, column j0
        use_nearest = (~mask_f.unsqueeze(1)) | (total == 0)
        return torch.where(use_nearest, nearest, w / total.clamp_min(1e-12))

    def forward(self, x, xyz_c, mask_c, xyz_f, mask_f):
        w = self.weights(xyz_c, mask_c, xyz_f, mask_f).to(x.dtype)
        out = None
        for i, (dr, j) in enumerate(self._candidates(x.shape[-1], x.device)):
            term = self._shift_rows(x, dr).index_select(-1, j) * w[:, i:i + 1]
            out = term if out is None else out + term
        return out


class DecoderStage(nn.Module):
    """[x2 upsample] -> concat skip -> conv-bn-relu -> residual block."""

    def __init__(self, c_in, c_skip, c_out, upsample, bn_d, rows=3, k=4):
        super().__init__()
        self.upsample = upsample
        if upsample == "trilinear":
            self.up = TrilinearUpsample(rows, k)
        elif upsample == "transpose":
            self.up = nn.ConvTranspose2d(c_in, c_in, kernel_size=(1, 4),
                                         stride=(1, 2), padding=(0, 1))
        else:
            self.up, c_skip = None, 0
        self.conv = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False)),
            ("bn", nn.BatchNorm2d(c_out, momentum=bn_d)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))
        self.block = BasicBlock(c_out, [c_in, c_out], bn_d)

    def forward(self, x, skip=None, coarse=None, fine=None):
        if self.upsample == "trilinear":
            x = self.up(x, coarse[0], coarse[1], fine[0], fine[1])
        elif self.upsample == "transpose":
            x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(self.conv(x))


class Decoder(nn.Module):
    def __init__(self, params, backbone_strides, skip_channels, feature_depth):
        super().__init__()
        channels = params["channels"]
        # the decoder undoes the backbone strides in reverse order
        ups = [s == 2 for s in reversed(backbone_strides)]
        assert len(ups) == len(channels)
        self.ups = ups
        stages, c_in = [], feature_depth
        os = 2 ** sum(ups)
        for up, c_out in zip(ups, channels):
            if up:
                os //= 2
            stages.append(DecoderStage(
                c_in, skip_channels.get(os, 0), c_out,
                params["upsample"] if up else None, params["bn_d"],
                params.get("trilinear_rows", 3), params.get("trilinear_k", 4)))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.dropout = nn.Dropout2d(params["dropout"])
        self.out_channels = channels[-1]

    def forward(self, x, skips, pyramid):
        os = 2 ** sum(self.ups)
        for up, stage in zip(self.ups, self.stages):
            if up:
                coarse = pyramid[os]
                os //= 2
                x = stage(x, skips[os], coarse, pyramid[os])
            else:
                x = stage(x)
        return self.dropout(x)


class ErrorHead(nn.Module):
    """Per-pixel non-negative error radius (Sec. III-C, three conv blocks)."""

    def __init__(self, c_in, bn_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_in, momentum=bn_d), nn.LeakyReLU(0.1),
            nn.Conv2d(c_in, c_in, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_in, momentum=bn_d), nn.LeakyReLU(0.1),
            nn.Conv2d(c_in, 1, 1),
        )

    def forward(self, x):
        return F.softplus(self.net(x))


class LPSAD(nn.Module):
    def __init__(self, cfg: dict, nclasses: int, ncategories: int):
        super().__init__()
        m = cfg["model"]
        b = m["backbone"]
        self.backbone = DarknetBackbone({
            "input_depth": b["input_depth"], "dropout": b["dropout"],
            "bn_d": b["bn_d"], "OS": b["OS"], "extra": {"layers": b["layers"]},
        })
        strides = self.backbone.strides
        # channels of the skip stored at each output stride
        widths = [32, 64, 128, 256, 512]
        skip_channels, os = {}, 1
        for s, c in zip(strides, widths):
            if s == 2:
                skip_channels[os] = c
                os *= 2

        def decoder():
            return Decoder(m["decoder"], strides, skip_channels,
                           self.backbone.get_last_depth())

        self.sem_decoder = decoder()
        self.inst_decoder = decoder()
        c = self.sem_decoder.out_channels
        bn_d = m["decoder"]["bn_d"]

        # semantic decoder: embedding -> classes, categories; embedding radius
        self.embedding = nn.Conv2d(c, m["embedding_dim"], 1)
        self.class_head = nn.Conv2d(m["embedding_dim"], nclasses, 1)
        self.category_head = nn.Conv2d(m["embedding_dim"], ncategories, 1)
        self.embedding_err = ErrorHead(c, bn_d)
        # instance decoder: 3D offset to the instance centre, offset radius
        self.offset_head = nn.Conv2d(c, 3, 1)
        self.offset_err = ErrorHead(c, bn_d)

    def forward(self, image, xyz, mask):
        pyramid = point_pyramid(xyz, mask, self.backbone.strides)
        feats, skips = self.backbone(image)
        sem = self.sem_decoder(feats, skips, pyramid)
        inst = self.inst_decoder(feats, skips, pyramid)
        emb = self.embedding(sem)
        return {
            "logits": self.class_head(emb),
            "cat_logits": self.category_head(emb),
            "embedding": emb,
            "embedding_err": self.embedding_err(sem),
            "offset": self.offset_head(inst),
            "offset_err": self.offset_err(inst),
        }

    def load_rangenet_backbone(self, path: str | Path):
        """Initialise the encoder from a RangeNet++ semantic model directory."""
        state = torch.load(Path(path) / "backbone", map_location="cpu",
                           weights_only=True)
        self.backbone.load_state_dict(state, strict=True)

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)

    def freeze_backbone_bn(self):
        """Keep backbone BatchNorm statistics fixed (call after .train())."""
        for mod in self.backbone.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
