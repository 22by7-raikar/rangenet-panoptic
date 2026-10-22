"""LPSAD losses (paper Eqs. 2-7) plus the category loss."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LabelSpace


def frequency_weights(content: np.ndarray, c: float, ignore: list[int]) -> torch.Tensor:
    """w = 1 / ln(c + f)  (Eq. 4; the ln(c + f) form keeps it positive)."""
    w = 1.0 / np.log(c + content)
    w[ignore] = 0.0
    return torch.tensor(w, dtype=torch.float32)


def weighted_ce(logits, target, weight):
    """Cross-entropy normalised by the target weights; 0 if all ignored."""
    nll = F.cross_entropy(logits.float(), target, reduction="none")
    w = weight[target]
    return (w * nll).sum() / w.sum().clamp_min(1e-6)


def segment_mean(values, index, n):
    """Mean of rows of ``values`` per segment ``index`` (0..n-1)."""
    out = values.new_zeros((n,) + values.shape[1:])
    out.index_add_(0, index, values)
    count = torch.bincount(index, minlength=n).clamp_min(1).to(values.dtype)
    return out / count.view(-1, *([1] * (values.dim() - 1)))


class LPSADLoss(nn.Module):
    def __init__(self, cfg: dict, labels: LabelSpace):
        super().__init__()
        self.w = cfg["loss"]
        c = self.w["class_weight_c"]
        self.register_buffer("class_w", frequency_weights(labels.content, c, labels.ignore))
        cat_content = np.zeros(labels.ncategories)
        np.add.at(cat_content, labels.class_to_category, labels.content)
        self.register_buffer("cat_w", frequency_weights(cat_content, c, [0]))

    def forward(self, out: dict, batch: dict):
        mask = batch["mask"]
        terms = {
            "sem": weighted_ce(out["logits"], batch["sem"], self.class_w),
            "cat": weighted_ce(out["cat_logits"], batch["cat"], self.cat_w),
        }
        stats = {}

        inst = batch["inst"]
        valid = (inst > 0) & mask
        zero = out["offset"].sum() * 0.0
        if valid.any():
            b = torch.arange(inst.shape[0], device=inst.device).view(-1, 1, 1).expand_as(inst)
            gid = b[valid] * (int(inst.max()) + 1) + inst[valid]
            _, idx = torch.unique(gid, return_inverse=True)
            n = int(idx.max()) + 1
            scan_of = torch.zeros(n, dtype=torch.long, device=idx.device)
            scan_of[idx] = b[valid]

            def pix(t):  # (B, C, H, W) -> (N_valid, C)
                return t.permute(0, 2, 3, 1)[valid].float()

            # Eq. 2: offset to the instance centre, averaged per instance
            off_err = pix(out["offset"]) - pix(batch["offset"])
            sq = (off_err ** 2).sum(1)
            terms["center"] = segment_mean(sq, idx, n).mean()
            # Eq. 3: learn the (relative) offset error as a radius
            err = sq.detach().sqrt()
            r = pix(out["offset_err"]).squeeze(1)
            terms["center_err"] = segment_mean((r - err) ** 2, idx, n).mean()

            # Eqs. 5-6: discriminative loss on the semantic embedding
            emb = pix(out["embedding"])
            mean_emb = segment_mean(emb, idx, n)
            d_in = ((emb - mean_emb[idx]) ** 2).sum(1)
            terms["attract"] = segment_mean(d_in, idx, n).mean()
            d_between = torch.cdist(mean_emb, mean_emb) ** 2
            pairs = (scan_of[:, None] == scan_of[None, :]) & ~torch.eye(n, dtype=torch.bool, device=idx.device)
            if pairs.any():
                terms["repel"] = (1.0 / (d_between[pairs] + self.w["repel_delta"])).mean()
            else:
                terms["repel"] = zero
            # Eq. 7: embedding radius
            e_err = d_in.detach().sqrt()
            re = pix(out["embedding_err"]).squeeze(1)
            terms["embed_err"] = segment_mean((re - e_err) ** 2, idx, n).mean()

            stats = {"center_err_mean": err.mean().item(),
                     "embed_err_mean": e_err.mean().item()}
        else:
            for k in ("center", "center_err", "attract", "repel", "embed_err"):
                terms[k] = zero

        weights = {"sem": "w_sem", "cat": "w_cat", "center": "w_center",
                   "center_err": "w_center_err", "attract": "w_attract",
                   "repel": "w_repel", "embed_err": "w_embed_err"}
        total = sum(self.w[weights[k]] * v for k, v in terms.items())
        return total, {k: v.detach() for k, v in terms.items()}, stats
