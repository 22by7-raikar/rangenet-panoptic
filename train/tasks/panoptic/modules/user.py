"""Panoptic inference wrapper around the pretrained semantic RangeNet++ model."""

import os
import time

import numpy as np
import torch

from tasks.panoptic.postproc import (PanopticConfig, make_clusterer,
                                     build_projection_index, pack_panoptic)
from tasks.semantic.modules.user import User


class PanopticUser(User):
  """Run semantic inference and write packed SemanticKITTI panoptic labels."""

  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,
               panoptic_config=None, instance_method="range-image"):
    super().__init__(ARCH, DATA, datadir, logdir, modeldir)
    if self.post is not None:
      raise ValueError("Disable semantic KNN post-processing for panoptic inference. "
                       "The panoptic layer performs its own projection-aware "
                       "backprojection.")
    self.clusterer = make_clusterer(
        instance_method,
        panoptic_config if panoptic_config is not None else PanopticConfig())

  def infer_subset(self, loader, to_orig_fn):
    self.model.eval()
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      previous_end = time.time()
      for _, batch in enumerate(loader):
        (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y,
         _, _, _, unproj_xyz, _, _, npoints) = batch
        point_count = int(npoints[0])
        path_seq = path_seq[0]
        path_name = path_name[0]
        p_x = p_x[0, :point_count]
        p_y = p_y[0, :point_count]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()

        inference_start = time.time()
        proj_output = self.model(proj_in, proj_mask)[0]
        if self.gpu:
          torch.cuda.synchronize()
        inference_seconds = time.time() - inference_start

        # Network output has shape [classes, H, W]. Indexing with each input
        # point's projected coordinates matches the semantic inference path.
        point_probabilities = proj_output[:, p_y, p_x].transpose(0, 1)
        point_confidence, point_semantic = point_probabilities.max(dim=1)
        learning_semantic = point_semantic.cpu().numpy().astype(np.int32)
        confidence = point_confidence.cpu().numpy().astype(np.float32)
        points = unproj_xyz[0, :point_count].cpu().numpy().astype(np.float32)
        projection = build_projection_index(
            points, p_x.cpu().numpy(), p_y.cpu().numpy(),
            image_shape=(proj_output.shape[1], proj_output.shape[2]))

        cluster_start = time.time()
        instances = self.clusterer.cluster(
            points, learning_semantic, confidence, projection,
            p_x.cpu().numpy(), p_y.cpu().numpy())
        clustering_seconds = time.time() - cluster_start

        # Evaluation expects original SemanticKITTI ids, while clustering used
        # the model's contiguous learning ids (1..8 are the thing classes).
        original_semantic = to_orig_fn(learning_semantic).astype(np.uint32)
        packed = pack_panoptic(original_semantic, instances)
        output_path = os.path.join(self.logdir, "sequences", path_seq,
                                   "predictions", path_name)
        packed.astype(np.uint32).tofile(output_path)

        elapsed = time.time() - previous_end
        print("Panoptic seq", path_seq, "scan", path_name,
              "network={:.3f}s".format(inference_seconds),
              "cluster={:.3f}s".format(clustering_seconds),
              "total={:.3f}s".format(elapsed))
        previous_end = time.time()
