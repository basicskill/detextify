import os
from detextify.utils import TextBox
from paddleocr import PaddleOCR
from typing import Sequence
import numpy as np

import logging
paddle_logger = logging.getLogger("ppocr").setLevel(logging.ERROR)


# This class is separate from `text_detector.py` because it depends on the paddle library, which has many wheels
# (see https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html).
# We'll ask users to manually install the right version for their system. However, for those users who don't want to
# use/install paddle, we want `from detextify import text_detector` to work (i.e., not fail on `import paddleocr`).


class PaddleTextDetector:
  """Uses PaddleOCR for text detection: https://github.com/PaddlePaddle/PaddleOCR"""

  def __init__(self, pad_size: int = 30, base_dir: str = None):
    """
    Args:
      pad_size: Empirically, Paddle returns text boxes that are quite tight. When we mask out the text boxes and the
        text is not masked out completely, the in-painting models regenerate text. The padding adds a safety margin to
        the text boxes to prevent such situations.
    """
    kwargs = {"lang": "en"}
    if base_dir is not None:
      self.base_dir = os.path.abspath(base_dir)
      kwargs.update({
        "det_model_dir": os.path.join(self.base_dir, "whl", "det", "en"),
        "rec_model_dir": os.path.join(self.base_dir, "whl", "rec", "en"),
        "cls_model_dir": os.path.join(self.base_dir, "whl", "cls", "en"),
        "layout_model_dir": os.path.join(self.base_dir, "whl", "layout", "en"),
        "table_model_dir": os.path.join(self.base_dir, "whl", "table", "en"),
      })

    self.ocr = PaddleOCR(**kwargs)
    self.pad_size = pad_size

  def detect_text(self, image: np.ndarray) -> Sequence[TextBox]:
    result = self.ocr.ocr(image, cls=True)[0]
    text_boxes = []

    for line in result:
      points = line[0]
      text = line[1][0]

      # These points are not necessarily a rectangle, but rather a polygon.
      # We'll find the smallest enclosing rectangle.
      ys = [point[0] for point in points]
      xs = [point[1] for point in points]
      
      tl_y = min(ys)
      tl_x = min(xs)
      h = max(ys) - tl_y
      w = max(xs) - tl_x

      if h < 0 or w < 0:
        paddle_logger.error(f"Malformed bounding box from Paddle: {points}")

      if self.pad_size:
          image_height, image_width = image.shape[:2]

          tl_y = max(0, tl_y - self.pad_size)
          tl_x = max(0, tl_x - self.pad_size)

          h = min(h + self.pad_size, image_height - tl_y)
          w = min(w + self.pad_size, image_width - tl_x)

      text_boxes.append(TextBox(int(tl_y), int(tl_x), int(h), int(w), text))
    return text_boxes

