import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd

from src.scripts import tricam_offline as tricam


class FakeCapture:
    def __init__(self, readable=True):
        self.readable = readable

    def set(self, key, value):
        self.frame = value

    def read(self):
        # BGR blue must become RGB blue for classifier input.
        return self.readable, np.full((40, 60, 3), (255, 0, 0), dtype=np.uint8)


def single_tracks(frames):
    return [pd.DataFrame([[frame, 1, 5, 5, 40, 35, 0.9]],
                         columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
            for frame in frames]


def predictor():
    model = Mock()
    model.enabled = ["major_damage", "wrong_brand_or_model"]
    model.predict.return_value = {"bottle_id": 1, "status": "NOK", "reasons": ["major_damage"]}
    return model


class TricamClassifierTests(unittest.TestCase):
    def test_matched_views_use_rgb_and_only_one_classifier_call(self):
        frames = [12, 9, 10]
        model = predictor()
        ocr = Mock()
        ocr.read.return_value = {"readings": [], "weight_kg_candidates": [], "year_candidates": []}
        predictions = tricam.inspect_crossings(
            model, [FakeCapture() for _ in range(3)], single_tracks(frames),
            [[(1, f)] for f in frames], [2, -1, 0], 0, 20, ocr=ocr)
        self.assertEqual(predictions[1]["ready_frame"], 10)
        model.predict.assert_called_once()
        crops = model.predict.call_args.args[0]
        self.assertEqual(set(crops), {"front", "back_left", "back_right"})
        self.assertTrue(all(image.getpixel((0, 0)) == (0, 0, 255) for image in crops.values()))
        self.assertEqual(ocr.read.call_count, 3)
        for call, camera in zip(ocr.read.call_args_list, tricam.VIDEO_CAMERAS):
            self.assertIs(call.args[0], crops[camera])
        self.assertEqual(set(predictions[1]["ocr"]), set(tricam.VIDEO_CAMERAS))
        self.assertEqual(tricam.inspection_counts(predictions, 9, 0)["NOK"], 0)
        self.assertEqual(tricam.inspection_counts(predictions, 10, 0)["NOK"], 1)
        self.assertEqual(tricam.inspection_counts(predictions, 11, 0)["NOK"], 1)

    def test_mismatched_times_or_unreadable_crop_do_not_produce_ok(self):
        model = predictor()
        ocr = Mock()
        for frames, captures in [([10, 30, 10], [FakeCapture()] * 3),
                                  ([10] * 3, [FakeCapture(), FakeCapture(False), FakeCapture()])]:
            results = tricam.inspect_crossings(model, captures, single_tracks(frames),
                                               [[(1, f)] for f in frames], [0] * 3, 0, 40, ocr=ocr)
            self.assertEqual(results[1]["status"], "INCOMPLETE")
        model.predict.assert_not_called()
        ocr.read.assert_not_called()

    def test_different_camera_counts_stop_classification(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            tricam.inspect_crossings(predictor(), [], [], [[(1, 10)], [], [(1, 10)]], [], 0, 40)

    def test_clip_counts_exclude_bottles_completed_before_clip(self):
        results = {1: {"status": "OK", "ready_frame": 9},
                   2: {"status": "NOK", "ready_frame": 11},
                   3: {"status": "INCOMPLETE", "ready_frame": 12}}
        self.assertEqual(tricam.inspection_counts(results, 12, 10),
                         {"OK": 0, "NOK": 1, "INCOMPLETE": 1})

    def test_video_overlay_and_counter_export_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = []
            for i, camera in enumerate(tricam.VIDEO_CAMERAS):
                video = root / f"{camera}.mp4"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 90))
                self.assertTrue(writer.isOpened())
                rows = []
                for frame in range(48):
                    writer.write(np.full((90, 160, 3), 70, dtype=np.uint8))
                    rows.append([frame, i + 1, -10 + frame * 3, 5, 10 + frame * 3, 80, 0.9])
                writer.release()
                pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]).to_csv(
                    video.with_suffix(".csv"), index=False)
                videos.append(video)
            weights = root / "detector.pt"
            checkpoint = root / "classifier.pt"
            weights.touch()
            checkpoint.touch()
            model = predictor()
            with patch("src.scripts.classifier.BottleClassifier", return_value=model):
                output = tricam.main(["--weights", str(weights), "--videos", *map(str, videos),
                                      "--classifier", str(checkpoint), "--tracks-dir", str(root),
                                      "--output", str(root / "result"), "--device", "cpu", "--no-display"])
            model.predict.assert_called_once()
            summary = json.loads(output.with_name("predictions.json").read_text())
            self.assertEqual(summary["counts"], {"OK": 0, "NOK": 1, "INCOMPLETE": 0})
            capture = cv2.VideoCapture(str(output))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 48)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 35)
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (720, 1280))
            # Red rejection annotations exist in the video tiles after the crossing.
            tiles = frame[:360]
            red = (tiles[:, :, 2] > 160) & (tiles[:, :, 1] < 120) & (tiles[:, :, 0] < 120)
            self.assertGreater(np.count_nonzero(red), 20)

            ocr = Mock()
            ocr.read.return_value = {"readings": [], "weight_kg_candidates": [13.4],
                                     "year_candidates": [2025]}
            with patch("src.scripts.ocr.BottleOCR", return_value=ocr) as reader:
                output = tricam.main(["--weights", str(weights), "--videos", *map(str, videos),
                                      "--ocr", "--tracks-dir", str(root),
                                      "--output", str(root / "ocr-result"), "--device", "cpu", "--no-display"])
            reader.assert_called_once_with("cpu")
            self.assertEqual(ocr.read.call_count, 3)
            summary = json.loads(output.with_name("predictions.json").read_text())
            self.assertTrue(summary["ocr_enabled"])
            self.assertIsNone(summary["checkpoint"])
            self.assertIsNone(summary["counts"])
            result = summary["predictions"][0]
            self.assertEqual(result["status"], "NOT_CHECKED")
            self.assertEqual(set(result["ocr"]), set(tricam.VIDEO_CAMERAS))
            self.assertEqual(result["ocr"]["front"]["year_candidates"], [2025])
            self.assertEqual(result["crops"]["front"]["track_id"], 3)
            capture = cv2.VideoCapture(str(output))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 48)
            capture.release()


if __name__ == "__main__":
    unittest.main()
