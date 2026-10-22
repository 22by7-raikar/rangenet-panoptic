"""Training / validation loop."""

import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cluster import panoptic_inference
from .config import LabelSpace
from .dataset import Augment, SemanticKittiPanoptic, collate
from .losses import LPSADLoss
from .metrics import PanopticEval
from .model import LPSAD


def make_loader(cfg, labels, split, train, max_scans=None):
    t = cfg["train"]
    augment = Augment(**t["augment"]) if train and t.get("augment") else None
    ds = SemanticKittiPanoptic(t["data_root"], t["split"][split], cfg, labels,
                               gt=True, augment=augment, max_scans=max_scans)
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=train,
                      num_workers=t["workers"], collate_fn=collate,
                      pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=t["workers"] > 0)


def autocast_context(mode: str, device: torch.device):
    if device.type != "cuda" or mode == "none":
        return nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


class Trainer:
    """Owns model, data, optimiser and checkpoints for one training run.

    See docs/architecture.md, "training".

    @param cfg     full configuration
    @param labels  label space from build_label_space()
    @param logdir  directory for checkpoints and TensorBoard logs
    @param resume  checkpoint to resume from, or None
    """

    def __init__(self, cfg: dict, labels: LabelSpace, logdir: str | Path,
                 resume: str | Path | None = None):
        self.cfg = cfg
        self.t = cfg["train"]
        self.labels = labels
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.t["seed"])
        np.random.seed(self.t["seed"])
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = LPSAD(cfg, labels.nclasses, labels.ncategories).to(self.device)
        if cfg["model"].get("pretrained"):
            self.model.load_rangenet_backbone(cfg["model"]["pretrained"])
            print(f"loaded RangeNet++ backbone from {cfg['model']['pretrained']}")
        self.forward = torch.compile(self.model) if self.t["compile"] else self.model
        self.criterion = LPSADLoss(cfg, labels).to(self.device)
        self.is_thing = torch.from_numpy(labels.is_thing()).to(self.device)

        self.train_loader = make_loader(cfg, labels, "train", train=True)
        self.val_loader = make_loader(cfg, labels, "valid", train=False,
                                      max_scans=self.t.get("val_max_scans"))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.t["lr"],
                                          betas=tuple(self.t["betas"]),
                                          weight_decay=self.t["weight_decay"])
        steps_per_epoch = max(1, len(self.train_loader) // self.t["accumulate"])
        warmup = self.t["warmup_epochs"] * steps_per_epoch

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            return self.t["lr_decay"] ** ((step - warmup) / steps_per_epoch)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(
            self.t["amp"] == "fp16" and self.device.type == "cuda"))

        self.epoch = 0
        self.best_pq = -1.0
        scale = cfg["cluster"]["radius_scale"]
        self.radius_scale = (1.0, 1.0) if scale == "auto" else (float(scale), float(scale))
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.logdir))
        except ImportError:
            pass
        if resume:
            self.load(resume)

    # ------------------------------------------------------------------ io
    def save(self, name: str, metrics: dict | None = None):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "best_pq": self.best_pq,
            "radius_scale": self.radius_scale,
            "config": self.cfg,
            "metrics": metrics,
        }, self.logdir / name)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.epoch = ckpt["epoch"]
        self.best_pq = ckpt["best_pq"]
        self.radius_scale = tuple(ckpt["radius_scale"])
        print(f"resumed from {path} at epoch {self.epoch}")

    def log(self, tag, values: dict, step):
        if self.writer is None:
            return
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{tag}/{k}", v, step)

    # --------------------------------------------------------------- train
    def train_epoch(self):
        """One pass over the training set; returns mean loss terms and errors."""
        self.model.train()
        frozen = self.epoch < self.t["freeze_backbone_epochs"]
        self.model.set_backbone_trainable(not frozen)
        if frozen or self.t["freeze_backbone_bn"]:
            self.model.freeze_backbone_bn()

        acc = self.t["accumulate"]
        sums, n, err_sum, err_n = {}, 0, np.zeros(2), 0
        start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(self.train_loader):
            batch = to_device(batch, self.device)
            with autocast_context(self.t["amp"], self.device):
                out = self.forward(batch["input"], batch["xyz"], batch["mask"])
            loss, terms, stats = self.criterion(out, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {terms}")
            self.scaler.scale(loss / acc).backward()

            if (i + 1) % acc == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.t["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
            if stats:
                err_sum += (stats["center_err_mean"], stats["embed_err_mean"])
                err_n += 1
            if (i + 1) % self.t["log_every"] == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                rate = (i + 1) / (time.time() - start)
                print(f"epoch {self.epoch} [{i + 1}/{len(self.train_loader)}] "
                      f"loss {loss.item():.4f} " +
                      " ".join(f"{k} {v.item():.3f}" for k, v in terms.items()) +
                      f" lr {lr:.2e} {rate:.2f} it/s", flush=True)

        means = {k: v / max(n, 1) for k, v in sums.items()}
        means["center_err_mean"], means["embed_err_mean"] = (err_sum / max(err_n, 1)).tolist()
        return means

    # ------------------------------------------------------------ validate
    @torch.no_grad()
    def validate(self, train_errors: dict | None = None):
        """Validation losses and panoptic metrics; updates the radius scale.

        @param train_errors  mean train errors from train_epoch(), for the
                             radius scale (val error / train error)
        @return metrics from PanopticEval.results() plus ``loss_*`` terms
        """
        self.model.eval()
        ev = PanopticEval(self.labels.nclasses, self.labels.ignore, self.labels.things)
        sums, n, err_sum, err_n = {}, 0, np.zeros(2), 0
        # clustering uses the radius scale from the previous validation; the
        # scale is re-estimated below from this pass, so it lags one epoch.
        gen = torch.Generator().manual_seed(0)
        for batch in self.val_loader:
            batch = to_device(batch, self.device)
            with autocast_context(self.t["amp"], self.device):
                out = self.forward(batch["input"], batch["xyz"], batch["mask"])
            out = {k: v.float() for k, v in out.items()}
            _, terms, stats = self.criterion(out, batch)
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
            if stats:
                err_sum += (stats["center_err_mean"], stats["embed_err_mean"])
                err_n += 1
            preds = panoptic_inference(out, batch, self.is_thing, self.cfg["cluster"],
                                       self.radius_scale, gen)
            truth = zip(batch["point_sem"], batch["point_inst"], strict=True)
            for (sem, inst), (gs, gi) in zip(preds, truth, strict=True):
                ev.add(sem.cpu().numpy(), inst.cpu().numpy(), gs.numpy(), gi.numpy())

        val_err = err_sum / max(err_n, 1)
        if self.cfg["cluster"]["radius_scale"] == "auto":
            if train_errors and train_errors["center_err_mean"] > 0:
                # paper: scale learned radii by (val error / train error)
                self.radius_scale = (
                    float(val_err[0] / train_errors["center_err_mean"]),
                    float(val_err[1] / max(train_errors["embed_err_mean"], 1e-8)))
        else:
            s = float(self.cfg["cluster"]["radius_scale"])
            self.radius_scale = (s, s)
        res = ev.results()
        res.update({f"loss_{k}": v / max(n, 1) for k, v in sums.items()})
        return res

    # ----------------------------------------------------------------- fit
    def fit(self):
        """Train until ``train.epochs``, validating every ``val_every`` epochs."""
        while self.epoch < self.t["epochs"]:
            train = self.train_epoch()
            self.log("train", train, self.epoch)
            print(f"epoch {self.epoch} train: " +
                  " ".join(f"{k} {v:.4f}" for k, v in train.items()), flush=True)
            metrics = None
            if (self.epoch + 1) % self.t["val_every"] == 0:
                metrics = self.validate(train)
                self.log("valid", metrics, self.epoch)
                print(f"epoch {self.epoch} valid: PQ {metrics['PQ']:.4f} "
                      f"PQ_th {metrics['PQ_th']:.4f} PQ_st {metrics['PQ_st']:.4f} "
                      f"RQ {metrics['RQ']:.4f} SQ {metrics['SQ']:.4f} "
                      f"mIoU {metrics['mIoU']:.4f} radius_scale "
                      f"({self.radius_scale[0]:.2f}, {self.radius_scale[1]:.2f})",
                      flush=True)
            self.epoch += 1
            self.save("last.pt", metrics)
            if metrics and metrics["PQ"] > self.best_pq:
                self.best_pq = metrics["PQ"]
                self.save("best.pt", metrics)
                print(f"new best PQ {self.best_pq:.4f}", flush=True)
            if math.isnan(train.get("sem", 0.0)):
                raise FloatingPointError("training diverged")
