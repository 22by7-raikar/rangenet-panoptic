#!/usr/bin/env python3
"""Run pretrained RangeNet++ semantic inference with panoptic post-processing."""

import argparse
import datetime
import os
import shutil
import subprocess

import yaml

from tasks.panoptic.modules.user import PanopticUser
from tasks.panoptic.postproc import ClusterParameters, PanopticConfig


def _load_panoptic_config(path):
  if path is None:
    return PanopticConfig()
  with open(path, "r") as config_file:
    values = yaml.safe_load(config_file) or {}
  default = ClusterParameters(**values.get("default", {}))
  class_parameters = {
      int(label): ClusterParameters(**parameters)
      for label, parameters in values.get("class_parameters", {}).items()
  }
  return PanopticConfig(
      thing_classes=tuple(values.get("thing_classes", range(1, 9))),
      confidence_threshold=float(values.get("confidence_threshold", 0.0)),
      connectivity=int(values.get("connectivity", 4)),
      default=default,
      class_parameters=class_parameters)


def _make_output_directories(logdir, data_config):
  if os.path.isdir(logdir):
    raise FileExistsError("Refusing to overwrite existing output directory: {}".format(logdir))
  os.makedirs(os.path.join(logdir, "sequences"))
  sequences = set(data_config["split"]["train"] +
                  data_config["split"]["valid"] +
                  data_config["split"]["test"])
  for sequence in sequences:
    sequence = "{0:02d}".format(int(sequence))
    os.makedirs(os.path.join(logdir, "sequences", sequence, "predictions"))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", "-d", required=True,
                      help="SemanticKITTI root containing sequences/")
  parser.add_argument("--model", "-m", required=True,
                      help="Pretrained semantic RangeNet++ model directory")
  parser.add_argument("--log", "-l",
                      default=os.path.expanduser("~") + "/logs/panoptic-" +
                      datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                      help="New directory for packed panoptic predictions")
  parser.add_argument("--panoptic-config", default=None,
                      help="Optional YAML overriding clustering parameters")
  parser.add_argument("--instance-method", choices=("range-image", "euclidean"),
                      default="range-image",
                      help="Instance extractor; range-image is the main method")
  arguments = parser.parse_args()

  with open(os.path.join(arguments.model, "arch_cfg.yaml")) as config_file:
    architecture = yaml.safe_load(config_file)
  with open(os.path.join(arguments.model, "data_cfg.yaml")) as config_file:
    data = yaml.safe_load(config_file)
  _make_output_directories(arguments.log, data)
  print("Commit hash:", subprocess.check_output(
      ["git", "rev-parse", "--short", "HEAD"]).decode().strip())

  user = PanopticUser(architecture, data, arguments.dataset, arguments.log,
                      arguments.model, _load_panoptic_config(arguments.panoptic_config),
                      arguments.instance_method)
  user.infer()


if __name__ == "__main__":
  main()
