import unittest

import cv2
import numpy as np

from src.scripts import evaluate_recognition as recognition


class RecognitionTests(unittest.TestCase):
    def test_rotated_box_is_deskewed_before_cropping(self):
        # A white box rotated 30 degrees clockwise about its top-left, as Label Studio saves it.
        value = {"x": 30, "y": 20, "width": 40, "height": 15, "rotation": 30}
        image = np.zeros((200, 200, 3), np.uint8)
        cv2.fillPoly(image, [np.round(recognition.corners(value, 200, 200)).astype(np.int32)], (255, 255, 255))
        enclosing = recognition.region(image, value, deskew=False)
        straight = recognition.region(image, value, deskew=True)
        self.assertGreater((straight > 127).mean(), 0.7)
        self.assertLess((enclosing > 127).mean(), 0.6)
        self.assertEqual(straight.shape[:2], (30 + 2 * recognition.PAD, 80 + 2 * recognition.PAD))

    def test_text_comparison(self):
        self.assertEqual(recognition.edit_distance("TARE10.7", "TARE10,7"), 1)
        self.assertEqual(recognition.normal("tare 10.7 kg"), "TARE10.7KG")


if __name__ == "__main__":
    unittest.main()
