"""TensorBoard logger for the semantic task.

Keeps the interface of the original TensorFlow 1 logger
(https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514) on top of
``torch.utils.tensorboard``. Logging is skipped if tensorboard is missing.
"""

import numpy as np

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed
  SummaryWriter = None


class Logger(object):
  """Scalar, image and histogram summaries in one TensorBoard log directory."""

  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    if SummaryWriter is None:
      print("tensorboard not installed, not logging summaries")
      self.writer = None
    else:
      self.writer = SummaryWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    if self.writer:
      self.writer.add_scalar(tag, value, step)
      self.writer.flush()

  def image_summary(self, tag, images, step):
    """Log a list of HxWx3 (BGR, as produced by OpenCV) or HxW images."""
    if not self.writer:
      return
    for i, img in enumerate(images):
      img = np.asarray(img)
      if img.ndim == 3:
        self.writer.add_image('%s/%d' % (tag, i), img[..., ::-1].copy(), step,
                              dataformats='HWC')
      else:
        self.writer.add_image('%s/%d' % (tag, i), img, step, dataformats='HW')
    self.writer.flush()

  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""
    if self.writer:
      self.writer.add_histogram(tag, np.asarray(values), step, bins=bins)
      self.writer.flush()
