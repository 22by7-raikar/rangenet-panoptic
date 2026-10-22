"""End-to-end smoke test: train -> checkpoint -> infer -> evaluate."""

import subprocess
import sys

import numpy as np
import torch
from conftest import PANOPTIC

from lpsad.trainer import Trainer


def test_train_infer_evaluate(cfg, labels, tmp_path):
    torch.set_num_threads(2)
    logdir = tmp_path / "logs"
    trainer = Trainer(cfg, labels, logdir)
    trainer.fit()
    assert (logdir / "last.pt").exists() and (logdir / "best.pt").exists()
    ckpt = torch.load(logdir / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 2
    assert all(np.isfinite(s) and s > 0 for s in ckpt["radius_scale"])
    m = ckpt["metrics"]
    assert 0.0 <= m["PQ"] <= 1.0 and np.isfinite(m["loss_sem"])

    # resuming restores the epoch counter and keeps training
    cfg["train"]["epochs"] = 3
    resumed = Trainer(cfg, labels, logdir, resume=logdir / "last.pt")
    assert resumed.epoch == 2
    resumed.fit()
    assert torch.load(logdir / "last.pt", weights_only=False)["epoch"] == 3

    preds = tmp_path / "preds"
    root = cfg["train"]["data_root"]
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=PANOPTIC, check=True,
                                    capture_output=True, text=True)
    r = run("infer.py", "-m", str(logdir / "best.pt"), "-d", root, "-o", str(preds),
            "--split", "valid", "--workers", "0")
    files = sorted((preds / "sequences" / "08" / "predictions").glob("*.label"))
    assert len(files) == 4, r.stdout + r.stderr
    gt = np.fromfile(f"{root}/sequences/08/labels/000000.label", dtype=np.uint32)
    pred = np.fromfile(files[0], dtype=np.uint32)
    assert pred.shape == gt.shape
    assert set(np.unique(pred & 0xFFFF)) <= set(labels.inverse_lut.tolist())

    # evaluate.py reads the split from the config; point it at the toy data
    import yaml
    small = dict(yaml.safe_load(open(PANOPTIC / "config" / "lpsad-darknet53.yaml")))
    small["labels"] = str((PANOPTIC / "config" / small["labels"]).resolve())
    small["train"]["split"] = {"train": [0], "valid": [8], "test": [8]}
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(small))
    r = run("evaluate.py", "-d", root, "-p", str(preds), "-c", str(cfg_path))
    assert "evaluated 4 scans" in r.stdout and "PQ" in r.stdout


def test_overfits_one_batch(cfg, labels, tmp_path):
    """Sanity check that the losses are learnable."""
    from lpsad.dataset import SemanticKittiPanoptic, collate
    from lpsad.losses import LPSADLoss
    from lpsad.model import LPSAD

    torch.manual_seed(0)
    torch.set_num_threads(2)
    ds = SemanticKittiPanoptic(cfg["train"]["data_root"], [0], cfg, labels)
    batch = collate([ds[0], ds[1]])
    model = LPSAD(cfg, labels.nclasses, labels.ncategories)
    crit = LPSADLoss(cfg, labels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(30):
        out = model(batch["input"], batch["xyz"], batch["mask"])
        loss, terms, _ = crit(out, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or {k: v.item() for k, v in terms.items()}
    last = {k: v.item() for k, v in terms.items()}
    assert last["sem"] < 0.5 * first["sem"]
    assert last["center"] < 0.5 * first["center"]
