"""Training-free panoptic post-processing for projected LiDAR semantics.

The semantic RangeNet++ model predicts a class distribution for every pixel in
the spherical range image.  This module turns those predictions into panoptic
labels without changing or retraining the network: it joins compatible
neighbouring *thing* pixels in the range image and assigns an instance id to
each connected component.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


SEMANTIC_MASK = np.uint32(0xFFFF)


@dataclass(frozen=True)
class ClusterParameters:
  """Per-class geometry and component-size settings in learning-label space."""

  base_distance: float = 0.35
  range_scale: float = 0.015
  min_points: int = 1


@dataclass
class PanopticConfig:
  """Configuration for range-image connected-component instance extraction."""

  thing_classes: Sequence[int] = tuple(range(1, 9))
  confidence_threshold: float = 0.0
  connectivity: int = 4
  default: ClusterParameters = field(default_factory=ClusterParameters)
  class_parameters: Dict[int, ClusterParameters] = field(default_factory=dict)

  def parameters_for(self, semantic_id: int) -> ClusterParameters:
    return self.class_parameters.get(int(semantic_id), self.default)


class _UnionFind:
  def __init__(self, count: int):
    self.parent = np.arange(count, dtype=np.int32)
    self.rank = np.zeros(count, dtype=np.int8)

  def find(self, item: int) -> int:
    root = item
    while self.parent[root] != root:
      root = int(self.parent[root])
    while self.parent[item] != item:
      parent = int(self.parent[item])
      self.parent[item] = root
      item = parent
    return root

  def union(self, left: int, right: int) -> None:
    left_root = self.find(left)
    right_root = self.find(right)
    if left_root == right_root:
      return
    if self.rank[left_root] < self.rank[right_root]:
      left_root, right_root = right_root, left_root
    self.parent[right_root] = left_root
    if self.rank[left_root] == self.rank[right_root]:
      self.rank[left_root] += 1


def pack_panoptic(semantic_labels: np.ndarray,
                  instance_labels: np.ndarray) -> np.ndarray:
  """Pack SemanticKITTI panoptic labels: upper 16 bits instance, lower semantic."""
  semantic_labels = np.asarray(semantic_labels, dtype=np.uint32).reshape(-1)
  instance_labels = np.asarray(instance_labels, dtype=np.uint32).reshape(-1)
  if semantic_labels.shape != instance_labels.shape:
    raise ValueError("semantic and instance labels must have identical shape")
  if np.any(semantic_labels > SEMANTIC_MASK) or np.any(instance_labels > SEMANTIC_MASK):
    raise ValueError("SemanticKITTI labels use at most 16 bits per field")
  return (instance_labels << np.uint32(16)) | semantic_labels


def build_projection_index(points: np.ndarray, proj_x: np.ndarray,
                           proj_y: np.ndarray,
                           image_shape: Tuple[int, int]) -> np.ndarray:
  """Recreate RangeNet++'s nearest-point projection collision policy.

  The semantic dataloader exposes the projection coordinate of every original
  point but not its projected index image. This helper reconstructs that image
  from the original points. Far points are written first, so the closest point
  wins each pixel exactly as in ``LaserScan.do_range_projection``.
  """
  points = np.asarray(points, dtype=np.float32)
  proj_x = np.asarray(proj_x, dtype=np.int32).reshape(-1)
  proj_y = np.asarray(proj_y, dtype=np.int32).reshape(-1)
  if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError("points must have shape [N, 3]")
  if proj_x.shape[0] != points.shape[0] or proj_y.shape[0] != points.shape[0]:
    raise ValueError("projection coordinate arrays must have N entries")
  height, width = image_shape
  if height <= 0 or width <= 0:
    raise ValueError("image_shape must contain positive height and width")
  if (np.any(proj_x < 0) or np.any(proj_x >= width) or
      np.any(proj_y < 0) or np.any(proj_y >= height)):
    raise ValueError("projection coordinates are outside image_shape")

  projection = np.full((height, width), -1, dtype=np.int32)
  depth = np.linalg.norm(points, axis=1)
  for point_index in np.argsort(depth)[::-1]:
    projection[proj_y[point_index], proj_x[point_index]] = point_index
  return projection


class RangeImagePanopticClusterer:
  """Class-aware, range-adaptive connected components over a spherical image.

  ``proj_idx`` contains the source point retained at each range pixel after
  projection collisions. ``proj_x`` and ``proj_y`` map every original point
  back to a pixel, allowing the final instance ids to retain original scan
  ordering.
  """

  def __init__(self, config: PanopticConfig = None):
    self.config = config if config is not None else PanopticConfig()
    if self.config.connectivity not in (4, 8):
      raise ValueError("connectivity must be 4 or 8")
    self.thing_classes = frozenset(int(label) for label in self.config.thing_classes)

  def _neighbours(self, row: int, column: int,
                  height: int, width: int) -> Iterable[Tuple[int, int]]:
    # Horizontal indexing wraps because a spinning LiDAR observes a full circle.
    offsets = ((0, 1), (1, 0))
    if self.config.connectivity == 8:
      offsets += ((1, 1), (1, -1))
    for d_row, d_column in offsets:
      next_row = row + d_row
      if next_row < 0 or next_row >= height:
        continue
      yield next_row, (column + d_column) % width

  def _compatible(self, left_point: int, right_point: int,
                  semantic: np.ndarray, confidence: np.ndarray,
                  points: np.ndarray, ranges: np.ndarray) -> bool:
    label = int(semantic[left_point])
    if label not in self.thing_classes or label != int(semantic[right_point]):
      return False
    if (confidence[left_point] < self.config.confidence_threshold or
        confidence[right_point] < self.config.confidence_threshold):
      return False
    parameters = self.config.parameters_for(label)
    distance = float(np.linalg.norm(points[left_point] - points[right_point]))
    threshold = parameters.base_distance + parameters.range_scale * min(
        float(ranges[left_point]), float(ranges[right_point]))
    return distance <= threshold

  def cluster(self, points: np.ndarray, semantic_labels: np.ndarray,
              semantic_confidence: np.ndarray, proj_idx: np.ndarray,
              proj_x: np.ndarray, proj_y: np.ndarray) -> np.ndarray:
    """Return one instance id per input point, in original point-cloud order."""
    points = np.asarray(points, dtype=np.float32)
    semantic_labels = np.asarray(semantic_labels, dtype=np.int32).reshape(-1)
    semantic_confidence = np.asarray(semantic_confidence, dtype=np.float32).reshape(-1)
    proj_idx = np.asarray(proj_idx, dtype=np.int32)
    proj_x = np.asarray(proj_x, dtype=np.int32).reshape(-1)
    proj_y = np.asarray(proj_y, dtype=np.int32).reshape(-1)

    point_count = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
      raise ValueError("points must have shape [N, 3]")
    if any(values.shape[0] != point_count
           for values in (semantic_labels, semantic_confidence, proj_x, proj_y)):
      raise ValueError("all point-wise arrays must have N entries")
    if proj_idx.ndim != 2:
      raise ValueError("proj_idx must have shape [H, W]")
    if np.any(proj_idx >= point_count):
      raise ValueError("proj_idx contains an out-of-range point index")

    height, width = proj_idx.shape
    ranges = np.linalg.norm(points, axis=1)
    components = _UnionFind(point_count)

    for row in range(height):
      for column in range(width):
        left_point = int(proj_idx[row, column])
        if left_point < 0:
          continue
        for neighbour_row, neighbour_column in self._neighbours(
            row, column, height, width):
          right_point = int(proj_idx[neighbour_row, neighbour_column])
          if right_point >= 0 and self._compatible(
              left_point, right_point, semantic_labels, semantic_confidence,
              points, ranges):
            components.union(left_point, right_point)

    # Count retained projected points per component.  This avoids assigning an
    # instance to noise components smaller than their class-specific minimum.
    component_sizes: Dict[int, int] = {}
    for point_index in proj_idx[proj_idx >= 0]:
      point_index = int(point_index)
      if int(semantic_labels[point_index]) in self.thing_classes:
        root = components.find(point_index)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    component_ids: Dict[int, int] = {}
    next_instance_id = 1
    for point_index in proj_idx[proj_idx >= 0]:
      point_index = int(point_index)
      label = int(semantic_labels[point_index])
      if label not in self.thing_classes:
        continue
      root = components.find(point_index)
      if component_sizes[root] < self.config.parameters_for(label).min_points:
        continue
      if root not in component_ids:
        component_ids[root] = next_instance_id
        next_instance_id += 1

    # Back-project component ids. Points that collided at a range pixel use the
    # retained point's component, matching RangeNet++'s semantic backprojection.
    instance_labels = np.zeros(point_count, dtype=np.uint16)
    for point_index in range(point_count):
      label = int(semantic_labels[point_index])
      if label not in self.thing_classes:
        continue
      pixel_point = int(proj_idx[proj_y[point_index], proj_x[point_index]])
      if pixel_point >= 0:
        instance_labels[point_index] = component_ids.get(
            components.find(pixel_point), 0)
    return instance_labels


class EuclideanPanopticClusterer:
  """Fixed-radius 3D connected-component baseline.

  This intentionally ignores range-image adjacency and range scaling. It gives
  the project a conventional geometric baseline against which the spherical,
  range-adaptive method can be evaluated. A spatial hash avoids an O(N^2)
  neighbour search for large LiDAR scans.
  """

  def __init__(self, config: PanopticConfig = None):
    self.config = config if config is not None else PanopticConfig()
    self.thing_classes = frozenset(int(label) for label in self.config.thing_classes)

  def cluster(self, points: np.ndarray, semantic_labels: np.ndarray,
              semantic_confidence: np.ndarray, _proj_idx: np.ndarray,
              _proj_x: np.ndarray, _proj_y: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    semantic_labels = np.asarray(semantic_labels, dtype=np.int32).reshape(-1)
    semantic_confidence = np.asarray(semantic_confidence, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
      raise ValueError("points must have shape [N, 3]")
    if (semantic_labels.shape[0] != points.shape[0] or
        semantic_confidence.shape[0] != points.shape[0]):
      raise ValueError("all point-wise arrays must have N entries")

    point_count = points.shape[0]
    components = _UnionFind(point_count)
    eligible = ((semantic_confidence >= self.config.confidence_threshold) &
                np.isin(semantic_labels, list(self.thing_classes)))

    for semantic_id in self.thing_classes:
      indices = np.flatnonzero(eligible & (semantic_labels == semantic_id))
      if indices.size == 0:
        continue
      radius = self.config.parameters_for(semantic_id).base_distance
      if radius <= 0.0:
        raise ValueError("Euclidean radius must be positive")
      buckets: Dict[Tuple[int, int, int], list] = {}
      for point_index in indices:
        cell = tuple(np.floor(points[point_index] / radius).astype(np.int32))
        for dx in (-1, 0, 1):
          for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
              neighbour_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
              for neighbour_index in buckets.get(neighbour_cell, []):
                if np.linalg.norm(points[point_index] - points[neighbour_index]) <= radius:
                  components.union(int(point_index), int(neighbour_index))
        buckets.setdefault(cell, []).append(int(point_index))

    component_sizes: Dict[int, int] = {}
    for point_index in np.flatnonzero(eligible):
      root = components.find(int(point_index))
      component_sizes[root] = component_sizes.get(root, 0) + 1

    instance_labels = np.zeros(point_count, dtype=np.uint16)
    component_ids: Dict[int, int] = {}
    next_instance_id = 1
    for point_index in np.flatnonzero(eligible):
      point_index = int(point_index)
      root = components.find(point_index)
      parameters = self.config.parameters_for(int(semantic_labels[point_index]))
      if component_sizes[root] < parameters.min_points:
        continue
      if root not in component_ids:
        component_ids[root] = next_instance_id
        next_instance_id += 1
      instance_labels[point_index] = component_ids[root]
    return instance_labels


def make_clusterer(method: str, config: PanopticConfig = None):
  """Create one of the documented panoptic instance extraction methods."""
  if method == "range-image":
    return RangeImagePanopticClusterer(config)
  if method == "euclidean":
    return EuclideanPanopticClusterer(config)
  raise ValueError("Unknown panoptic instance method: {}".format(method))
