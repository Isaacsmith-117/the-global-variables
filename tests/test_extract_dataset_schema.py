import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.scripts import extract_dataset as extraction
from src.scripts.reason_schema import PROJECT_POLICY, REASONS, REASON_SCHEMA_VERSION


class ExtractionSchemaTests(unittest.TestCase):
    def test_real_extraction_keeps_four_reasons_unknown_even_for_verified_nok(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks_dir = root / "tracks"
            tracks_dir.mkdir()
            videos, matches = [], []
            for camera in ("front", "back_left", "back_right"):
                video = root / f"fixture_{camera}.avi"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (96, 80))
                self.assertTrue(writer.isOpened())
                rows = []
                for frame in range(60):
                    left = 6 + round(frame * 72 / 59)
                    image = np.zeros((80, 96, 3), dtype=np.uint8)
                    image[10:60, left:left + 12] = (100, 160, 30)
                    writer.write(image)
                    rows.append((frame, 1, left, 10, left + 12, 60, 0.9))
                writer.release()
                tracks = pd.DataFrame(rows, columns=("frame", "track_id", "x1", "y1", "x2", "y2", "conf"))
                tracks.to_csv(tracks_dir / f"{video.stem}.csv", index=False)
                crossing = extraction.crossings(tracks, 96)[0][1]
                matches.append({"recording_id": "fixture", "camera_id": camera,
                                "track_id": 1, "frame_index": crossing, "spreadsheet_id": 1})
                videos.append(video)
            truth = root / "fixture_ground_truth.csv"
            truth.write_text("id,primagaz_status,color\n1,NOK,black\n")
            match_path = root / "reviewed.csv"
            pd.DataFrame(matches).to_csv(match_path, index=False)

            for verified in (False, True):
                with self.subTest(verified=verified):
                    output = root / ("verified" if verified else "proposed")
                    args = ["--recording-id", "fixture", "--videos", *map(str, videos),
                            "--truth", str(truth), "--tracks-dir", str(tracks_dir), "--output", str(output)]
                    if verified:
                        args.extend(["--matches", str(match_path)])
                    extraction.main(args)
                    with (output / "manifest.csv").open() as manifest:
                        rows = list(csv.DictReader(manifest))
                    self.assertEqual(len(rows), 3)
                    self.assertEqual({r["camera_id"] for r in rows}, {"front", "back_left", "back_right"})
                    for row in rows:
                        self.assertEqual({reason: row[reason] for reason in REASONS}, dict.fromkeys(REASONS, "unknown"))
                        self.assertEqual(int(row["reason_schema_version"]), REASON_SCHEMA_VERSION)
                        self.assertEqual(row["project_policy"], PROJECT_POLICY)
                        self.assertEqual(row["primagaz_status"], "NOK" if verified else "")
                        self.assertTrue((output / row["image_path"]).is_file())
                    self.assertNotIn("major_damage", rows[0])
                    self.assertNotIn("heavy_dirt", rows[0])
                    self.assertNotIn("wrong_brand_or_model", rows[0])
                    run = json.loads((output / "run.json").read_text())
                    self.assertTrue(run["complete"])
                    self.assertEqual(run["reason_order"], list(REASONS))
                    self.assertEqual(run["reason_schema_version"], REASON_SCHEMA_VERSION)
                    self.assertEqual(run["project_policy"], PROJECT_POLICY)


if __name__ == "__main__":
    unittest.main()
