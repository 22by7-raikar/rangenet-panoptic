import unittest
import tempfile
from pathlib import Path

import numpy as np

from tasks.panoptic.postproc import (ClusterParameters, PanopticConfig,
                                     EuclideanPanopticClusterer,
                                     RangeImagePanopticClusterer,
                                     build_projection_index, pack_panoptic)
from common.laserscan import LaserScan
from tasks.panoptic.visualize import panoptic_colors, write_ply


class RangeImagePanopticClustererTest(unittest.TestCase):
  def setUp(self):
    config = PanopticConfig(
        thing_classes=(1,),
        confidence_threshold=0.5,
        default=ClusterParameters(base_distance=0.4, range_scale=0.0,
                                  min_points=1))
    self.clusterer = RangeImagePanopticClusterer(config)

  def test_wraps_across_spherical_seam(self):
    points = np.array([[10.0, 0.0, 0.0], [10.1, 0.0, 0.0]], np.float32)
    labels = np.array([1, 1])
    confidence = np.ones(2, np.float32)
    proj_idx = np.array([[0, -1, -1, 1]], np.int32)
    instances = self.clusterer.cluster(
        points, labels, confidence, proj_idx,
        proj_x=np.array([0, 3]), proj_y=np.array([0, 0]))
    self.assertEqual(instances.tolist(), [1, 1])

  def test_does_not_bridge_depth_discontinuity(self):
    points = np.array([[5.0, 0.0, 0.0], [5.8, 0.0, 0.0]], np.float32)
    labels = np.array([1, 1])
    confidence = np.ones(2, np.float32)
    proj_idx = np.array([[0, 1]], np.int32)
    instances = self.clusterer.cluster(
        points, labels, confidence, proj_idx,
        proj_x=np.array([0, 1]), proj_y=np.array([0, 0]))
    self.assertEqual(instances.tolist(), [1, 2])

  def test_low_confidence_pixel_breaks_component(self):
    points = np.array([[5.0, 0.0, 0.0], [5.1, 0.0, 0.0]], np.float32)
    labels = np.array([1, 1])
    confidence = np.array([0.9, 0.4], np.float32)
    proj_idx = np.array([[0, 1]], np.int32)
    instances = self.clusterer.cluster(
        points, labels, confidence, proj_idx,
        proj_x=np.array([0, 1]), proj_y=np.array([0, 0]))
    self.assertEqual(instances.tolist(), [1, 2])

  def test_stuff_has_no_instance_id(self):
    points = np.array([[5.0, 0.0, 0.0], [5.1, 0.0, 0.0]], np.float32)
    labels = np.array([9, 9])
    confidence = np.ones(2, np.float32)
    instances = self.clusterer.cluster(
        points, labels, confidence, np.array([[0, 1]], np.int32),
        proj_x=np.array([0, 1]), proj_y=np.array([0, 0]))
    self.assertEqual(instances.tolist(), [0, 0])

  def test_packs_semantic_kitti_label(self):
    packed = pack_panoptic(np.array([10, 40]), np.array([7, 0]))
    self.assertEqual(packed.tolist(), [(7 << 16) | 10, 40])

  def test_euclidean_baseline_joins_close_same_class_points(self):
    points = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0],
                       [2.0, 0.0, 0.0]], np.float32)
    baseline = EuclideanPanopticClusterer(self.clusterer.config)
    instances = baseline.cluster(
        points, np.array([1, 1, 1]), np.ones(3, np.float32),
        np.array([[-1]], np.int32), np.zeros(3), np.zeros(3))
    self.assertEqual(instances[0], instances[1])
    self.assertNotEqual(instances[0], instances[2])

  def test_projection_collision_keeps_nearest_point(self):
    points = np.array([[10.0, 0.0, 0.0], [2.0, 0.0, 0.0]], np.float32)
    projection = build_projection_index(
        points, proj_x=np.array([1, 1]), proj_y=np.array([0, 0]),
        image_shape=(1, 3))
    self.assertEqual(int(projection[0, 1]), 1)

  def test_existing_spherical_projection_drives_panoptic_clustering(self):
    # Use the exact projection implementation consumed by RangeNet++ rather
    # than an artificial index grid. The two close points form one car;
    # the distant point remains a separate car instance.
    points = np.array([[8.0, 0.0, 0.0], [8.1, -0.05, 0.0],
                       [8.0, 4.0, 0.0]], np.float32)
    scan = LaserScan(project=True, H=64, W=1024, fov_up=3.0, fov_down=-25.0)
    scan.set_points(points)
    projection = build_projection_index(
        scan.points, scan.proj_x, scan.proj_y, (scan.proj_H, scan.proj_W))
    instances = self.clusterer.cluster(
        scan.points, np.array([1, 1, 1]), np.ones(3, np.float32), projection,
        scan.proj_x, scan.proj_y)
    self.assertEqual(instances[0], instances[1])
    self.assertNotEqual(instances[0], instances[2])

  def test_visualizer_distinguishes_instances_and_writes_ply(self):
    labels = pack_panoptic(np.array([10, 10, 40]), np.array([1, 2, 0]))
    colors = panoptic_colors(labels, {10: [100, 150, 245], 40: [255, 0, 255]})
    self.assertFalse(np.array_equal(colors[0], colors[1]))
    self.assertTrue(np.array_equal(colors[2], np.array([255, 0, 255])))
    with tempfile.TemporaryDirectory() as temporary_directory:
      output = Path(temporary_directory) / "prediction.ply"
      write_ply(np.zeros((3, 3), np.float32), colors, str(output))
      self.assertIn("element vertex 3", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
  unittest.main()
