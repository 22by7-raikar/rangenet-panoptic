#!/usr/bin/env python3
"""Export a SemanticKITTI scan and packed panoptic labels as a colored PLY."""

import argparse
from pathlib import Path

import numpy as np
import yaml


def _instance_tint(instance_id: int) -> np.ndarray:
  """Return a deterministic bright RGB tint without a global random state."""
  value = (int(instance_id) * 2654435761) & 0xFFFFFFFF
  return np.array([(value >> 16) & 0x7F, (value >> 8) & 0x7F,
                   value & 0x7F], dtype=np.uint8)


def panoptic_colors(packed_labels: np.ndarray, color_map: dict) -> np.ndarray:
  """Color stuff by semantics and distinguish thing instances by a stable tint."""
  packed_labels = np.asarray(packed_labels, dtype=np.uint32).reshape(-1)
  semantic = packed_labels & np.uint32(0xFFFF)
  instances = packed_labels >> np.uint32(16)
  colors = np.zeros((packed_labels.shape[0], 3), dtype=np.uint8)
  for index, semantic_id in enumerate(semantic):
    # SemanticKITTI YAML colors are stored as BGR.
    bgr = color_map.get(int(semantic_id), (40, 40, 40))
    base = np.asarray(bgr[::-1], dtype=np.uint8)
    if instances[index] == 0:
      colors[index] = base
    else:
      colors[index] = (base.astype(np.uint16) * 2 +
                       _instance_tint(int(instances[index]))).astype(np.uint16) // 3
  return colors


def write_ply(points: np.ndarray, colors: np.ndarray, output_path: str) -> None:
  points = np.asarray(points, dtype=np.float32)
  colors = np.asarray(colors, dtype=np.uint8)
  if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
    raise ValueError("points and colors must both have shape [N, 3]")
  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8") as ply_file:
    ply_file.write("ply\nformat ascii 1.0\n")
    ply_file.write("element vertex {}\n".format(points.shape[0]))
    ply_file.write("property float x\nproperty float y\nproperty float z\n")
    ply_file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    ply_file.write("end_header\n")
    for point, color in zip(points, colors):
      ply_file.write("{:.5f} {:.5f} {:.5f} {} {} {}\n".format(
          point[0], point[1], point[2], int(color[0]), int(color[1]), int(color[2])))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scan", required=True, help="SemanticKITTI .bin scan")
  parser.add_argument("--prediction", required=True,
                      help="Packed uint32 SemanticKITTI panoptic .label file")
  parser.add_argument("--labels", required=True,
                      help="SemanticKITTI label YAML with color_map")
  parser.add_argument("--output", required=True, help="Output ASCII .ply path")
  arguments = parser.parse_args()

  points = np.fromfile(arguments.scan, dtype=np.float32).reshape((-1, 4))[:, :3]
  labels = np.fromfile(arguments.prediction, dtype=np.uint32)
  if points.shape[0] != labels.shape[0]:
    raise ValueError("scan has {} points but prediction has {} labels".format(
        points.shape[0], labels.shape[0]))
  with open(arguments.labels, encoding="utf-8") as yaml_file:
    color_map = yaml.safe_load(yaml_file)["color_map"]
  write_ply(points, panoptic_colors(labels, color_map), arguments.output)


if __name__ == "__main__":
  main()
