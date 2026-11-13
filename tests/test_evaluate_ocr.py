import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.scripts import evaluate_ocr
from src.scripts.ocr_selection import load_selection


def cache(folder, group, candidates):
    """A bottle cache with one view, in the format written by run_ocr_groups.py."""
    view = {"image_path": f"{group}_front.jpg", "proposals": [{}], "candidates": [
        {"field": field, "value": value, "field_confidence": field_score, "ocr_confidence": ocr_score}
        for field, value, field_score, ocr_score in candidates]}
    document = {"images": [view], "ocr_selection": load_selection(),
                "dataset_split_provenance": {"bottle_group": group}}
    (folder / f"{group}.json").write_text(json.dumps(document))


class EvaluateOCRTests(unittest.TestCase):
    def test_tuning_rejects_wrong_reads_and_reports_tare_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            caches = root / "cache"
            caches.mkdir()
            # b1 is read correctly with high confidence; b2's confident-looking read is wrong.
            cache(caches, "b1", [("tare_kg", 10.7, 0.9, 0.9), ("approval_until_date", "2034", 0.9, 0.8)])
            cache(caches, "b2", [("tare_kg", 15.0, 0.5, 0.4), ("approval_until_date", "2031", 0.6, 0.3)])
            cache(caches, "b3", [("tare_kg", 9.9, 0.9, 0.9)])
            references = root / "references.csv"
            with references.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["bottle_group", "workbook_tarra_kg", "tare_reference_eligible",
                                 "approval_until_year", "approval_until_reference_eligible"])
                writer.writerows([["b1", "10.65", "True", "2034", "True"], ["b2", "15.9", "True", "2030", "True"],
                                  ["b3", "", "False", "", "False"]])

            output = root / "report"
            evaluate_ocr.main([str(caches), "--references", str(references), "--output", str(output), "--tune"])
            report = json.loads((output / "metrics.json").read_text())
            tare = report["metrics"]["tare_kg"]
            self.assertEqual((tare["correct"], tare["wrong"], tare["abstained"]), (1, 0, 1))
            self.assertEqual(tare["accepted_without_reference"], 1)
            self.assertEqual(tare["best_read_max_abs_error_kg"], 0.9)
            self.assertEqual(report["metrics"]["approval_until_date"]["wrong"], 0)
            config = load_selection(output / "selection.validation.json")
            self.assertTrue(all(value is not None for limits in config["fields"].values() for value in limits.values()))
            for name in ("bottles.csv", "tuning_trials.csv", "outcomes.png", "tare_error.png", "confidence.png"):
                self.assertTrue((output / name).is_file(), name)

            frozen = root / "test-report"
            evaluate_ocr.main([str(caches), "--references", str(references), "--output", str(frozen),
                               "--config", str(output / "selection.validation.json")])
            self.assertEqual(json.loads((frozen / "metrics.json").read_text())["metrics"], report["metrics"])
            with self.assertRaises(SystemExit):
                evaluate_ocr.main([str(caches), "--references", str(references), "--output", str(frozen), "--tune"])


if __name__ == "__main__":
    unittest.main()
