"""Smoke test for the upstream RangeNet++ task: train one epoch, infer, evaluate.

Runs the task's own scripts on tiny synthetic scans on the CPU. Needs the full
requirements.txt (OpenCV, SciPy, ...). Run with
``python -m pytest train/tasks/semantic/tests``.
"""

import subprocess
import sys
from pathlib import Path

import yaml

SEMANTIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEMANTIC.parent / "panoptic" / "tests"))
from synthetic import write_dataset  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=SEMANTIC,
                       capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_train_infer_evaluate(tmp_path):
    data = write_dataset(tmp_path / "kitti", sequences=(0, 8, 11), scans_per_seq=2)
    arch = yaml.safe_load(open(SEMANTIC / "config/arch/darknet21.yaml"))
    arch["train"].update(max_epochs=1, batch_size=2, workers=0, save_scans=True,
                         save_summary=True)
    arch["dataset"]["sensor"]["img_prop"].update(width=256, height=16)
    labels = yaml.safe_load(open(SEMANTIC / "config/labels/semantic-kitti.yaml"))
    labels["split"] = {"train": [0], "valid": [8], "test": [11]}
    (tmp_path / "arch.yaml").write_text(yaml.safe_dump(arch))
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(labels))

    log, preds = tmp_path / "log", tmp_path / "preds"
    out = run("train.py", "-d", data, "-ac", tmp_path / "arch.yaml",
              "-dc", tmp_path / "data.yaml", "-l", log)
    assert "Finished Training" in out
    assert (log / "backbone").exists() and any((log / "tb").iterdir())

    run("infer.py", "-d", data, "-l", preds, "-m", log)
    assert len(list((preds / "sequences" / "08" / "predictions").glob("*.label"))) == 2

    out = run("evaluate_iou.py", "-d", data, "-p", preds, "--split", "valid",
              "-dc", tmp_path / "data.yaml")
    assert "IoU avg" in out
