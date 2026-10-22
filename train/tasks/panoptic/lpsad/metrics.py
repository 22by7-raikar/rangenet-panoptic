"""Point-level mIoU and panoptic quality, following semantic-kitti-api.

This mirrors the logic of ``PanopticEval`` in the official semantic-kitti-api
(``auxiliary/eval_np.py``): instance ids are shifted by one so that stuff
(instance 0) forms a single segment per class, points whose ground truth is
an ignored class are dropped, a segment pair matches at IoU > 0.5, and
unmatched segments with fewer than ``min_points`` points are not counted.
Use the official ``evaluate_panoptic.py`` for numbers you report.
"""

import numpy as np


class PanopticEval:
    def __init__(self, n_classes: int, ignore: list[int], things: list[int],
                 min_points: int = 50, offset: int = 2 ** 32):
        self.n = n_classes
        self.ignore = list(ignore)
        self.include = [c for c in range(n_classes) if c not in self.ignore]
        self.things = [c for c in things if c in self.include]
        self.stuff = [c for c in self.include if c not in self.things]
        self.min_points = min_points
        self.offset = offset
        self.reset()

    def reset(self):
        self.conf = np.zeros((self.n, self.n), dtype=np.int64)  # [pred, gt]
        self.tp = np.zeros(self.n, dtype=np.int64)
        self.fp = np.zeros(self.n, dtype=np.int64)
        self.fn = np.zeros(self.n, dtype=np.int64)
        self.iou_sum = np.zeros(self.n, dtype=np.float64)

    def add(self, pred_sem, pred_inst, gt_sem, gt_inst):
        pred_sem = np.asarray(pred_sem, dtype=np.int64)
        gt_sem = np.asarray(gt_sem, dtype=np.int64)
        np.add.at(self.conf, (pred_sem, gt_sem), 1)

        pred_inst = np.asarray(pred_inst, dtype=np.int64) + 1
        gt_inst = np.asarray(gt_inst, dtype=np.int64) + 1
        keep = ~np.isin(gt_sem, self.ignore)
        pred_sem, pred_inst = pred_sem[keep], pred_inst[keep]
        gt_sem, gt_inst = gt_sem[keep], gt_inst[keep]

        for cl in self.include:
            p = pred_inst * (pred_sem == cl)
            g = gt_inst * (gt_sem == cl)
            u_pred, c_pred = np.unique(p[p > 0], return_counts=True)
            u_gt, c_gt = np.unique(g[g > 0], return_counts=True)
            pred_idx = {k: i for i, k in enumerate(u_pred)}
            gt_idx = {k: i for i, k in enumerate(u_gt)}
            matched_pred = np.zeros(len(u_pred), dtype=bool)
            matched_gt = np.zeros(len(u_gt), dtype=bool)

            both = (p > 0) & (g > 0)
            combo, inter = np.unique(p[both] + self.offset * g[both], return_counts=True)
            if len(combo):
                gt_lab = combo // self.offset
                pred_lab = combo % self.offset
                gi = np.array([gt_idx[k] for k in gt_lab])
                pi = np.array([pred_idx[k] for k in pred_lab])
                iou = inter / (c_gt[gi] + c_pred[pi] - inter)
                hit = iou > 0.5
                self.tp[cl] += hit.sum()
                self.iou_sum[cl] += iou[hit].sum()
                matched_gt[gi[hit]] = True
                matched_pred[pi[hit]] = True
            self.fn[cl] += np.sum((c_gt >= self.min_points) & ~matched_gt)
            self.fp[cl] += np.sum((c_pred >= self.min_points) & ~matched_pred)

    def semantic_iou(self) -> np.ndarray:
        conf = self.conf.copy()
        conf[:, self.ignore] = 0
        tp = np.diag(conf).astype(np.float64)
        fp = conf.sum(1) - tp
        fn = conf.sum(0) - tp
        return tp / np.maximum(tp + fp + fn, 1e-15)

    def results(self) -> dict:
        eps = 1e-15
        sq = self.iou_sum / np.maximum(self.tp, eps)
        rq = self.tp / np.maximum(self.tp + 0.5 * self.fp + 0.5 * self.fn, eps)
        pq = sq * rq
        iou = self.semantic_iou()
        pq_dagger = pq.copy()
        pq_dagger[self.stuff] = iou[self.stuff]

        def mean(v, cls):
            return float(v[cls].mean()) if cls else 0.0

        return {
            "PQ": mean(pq, self.include), "PQ_dagger": mean(pq_dagger, self.include),
            "SQ": mean(sq, self.include), "RQ": mean(rq, self.include),
            "PQ_th": mean(pq, self.things), "SQ_th": mean(sq, self.things),
            "RQ_th": mean(rq, self.things),
            "PQ_st": mean(pq, self.stuff), "SQ_st": mean(sq, self.stuff),
            "RQ_st": mean(rq, self.stuff),
            "mIoU": mean(iou, self.include),
            "per_class": {"PQ": pq, "SQ": sq, "RQ": rq, "IoU": iou},
        }
