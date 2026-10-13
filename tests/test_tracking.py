import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from src.scripts import config, track


class TrackingTests(unittest.TestCase):
    def test_device_selection_for_different_computers(self):
        cases = ((True, False, "mps"), (False, True, "cuda:0"), (False, False, "cpu"))
        for mps, cuda, expected in cases:
            with (
                self.subTest(device=expected),
                patch.object(config.torch.backends.mps, "is_available", return_value=mps),
                patch.object(config.torch.cuda, "is_available", return_value=cuda),
            ):
                self.assertEqual(config.select_device(), expected)

    def test_bytetrack_recovers_same_id_after_brief_detection_gap(self):
        tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
            track_buffer=30, match_thresh=0.8, fuse_score=True,
        ))

        def detections(x, score=0.9):
            return Boxes(np.array([[x, 10, x + 20, 50, score, 0]], dtype=np.float32), (64, 64))

        first = tracker.update(detections(10))
        second = tracker.update(detections(11, score=0.2))
        missing = tracker.update(Boxes(np.empty((0, 6), dtype=np.float32), (64, 64)))
        recovered = tracker.update(detections(12))
        self.assertEqual(len(missing), 0)
        self.assertEqual(int(first[0, 4]), int(second[0, 4]))
        self.assertEqual(int(first[0, 4]), int(recovered[0, 4]))

    def test_missing_weights_do_not_start_tracking(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(track, "YOLO") as yolo:
            with self.assertRaisesRegex(SystemExit, "Trained weights missing"):
                track.main(["--weights", str(Path(directory) / "missing.pt"), "--source", "missing.mp4"])
            yolo.assert_not_called()

    def test_still_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, source = root / "best.pt", root / "image.jpg"
            weights.touch()
            source.touch()
            with self.assertRaisesRegex(SystemExit, "Source must be a video"):
                track.main(["--weights", str(weights), "--source", str(source)])

    def test_missing_video_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(track, "YOLO") as yolo:
            weights = Path(directory) / "best.pt"
            weights.touch()
            with self.assertRaisesRegex(SystemExit, "Video file missing"):
                track.main(["--weights", str(weights), "--source", str(Path(directory) / "missing.mp4")])
            yolo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
