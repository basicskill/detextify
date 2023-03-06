import numpy as np
from detextify.inpainter import Inpainter


class Detextifier:
    def __init__(self, text_detector, inpainter):
        self.text_detector = text_detector
        self.inpainter = inpainter

    def detextify(self, in_image: np.ndarray, prompt=Inpainter.DEFAULT_PROMPT, max_retries=5):
        out_image = in_image.copy()
        for i in range(max_retries):
            print(f"Iteration {i} of {max_retries} for image:")

            print(f"\tCalling text detector...")
            text_boxes = self.text_detector.detect_text(out_image)
            print(f"\tDetected {len(text_boxes)} text boxes.")

            if not text_boxes:
                break

            print(f"\tCalling in-painting model...")
            out_image = self.inpainter.inpaint(out_image, text_boxes, prompt)

        return out_image

