import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from src.scripts import compare_second_reads as compare
from src.scripts.ocr_selection import load_selection


class CompareSecondReadsTests(unittest.TestCase):
    def test_paired_comparison_counts_gains_losses_and_partial_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "b1_front.jpg"
            Image.new("RGB", (60, 40)).save(image)
            # Original: tare withheld (OCR confidence too low), year accepted from one view and one reader.
            view = {"image_path": str(image), "readings": [], "proposals": [
                {"field": "tare_kg", "box": [10, 10, 40, 30], "field_confidence": 0.9}], "candidates": [
                {"field": "tare_kg", "value": 10.0, "field_confidence": 0.9, "ocr_confidence": 0.3,
                 "box": [10, 10, 40, 30], "source": "field_region"},
                {"field": "approval_until_date", "value": "2034", "field_confidence": 0.9, "ocr_confidence": 0.9,
                 "box": [5, 5, 20, 15], "source": "field_region"}]}
            cache = root / "cache"
            cache.mkdir()
            (cache / "b1.json").write_text(json.dumps({"images": [view],
                                                       "dataset_split_provenance": {"bottle_group": "b1"}}))
            config = load_selection()
            for limits in config["fields"].values():
                limits.update(min_field_confidence=0.5, min_ocr_confidence=0.5, min_margin=0.0)
            (root / "selection.json").write_text(json.dumps(config))
            with (root / "references.csv").open("w", newline="") as file:
                csv.writer(file).writerows([["bottle_group", "workbook_tare_kg", "workbook_approval_year",
                                             "marking_tare_kg", "marking_approval_year"],
                                            ["b1", "10.7", "2034", "10.7", "2034"]])
            # PARSeq rereads: the tare box shows only "10 KG" (fraction dropped); the year box shows no year.
            predictor = Mock(side_effect=lambda crops: [("10 KG", 0.8)])
            doctr = Mock()
            doctr.models.recognition_predictor.return_value = predictor
            output = root / "comparison"
            arguments = [str(cache), "--config", str(root / "selection.json"), "--references",
                         str(root / "references.csv"), "--output", str(output)]
            with patch.dict(sys.modules, {"doctr": doctr, "doctr.models": doctr.models}):
                compare.main(arguments)
            report = json.loads((output / "metrics.json").read_text())["comparison"]
            for source in ("human", "workbook"):
                tare, year = report[source]["tare_kg"], report[source]["approval_until_date"]
                self.assertEqual((tare["original"]["blank"], tare["tare_reread"]["partial"]), (1, 1))
                self.assertEqual(tare["tare_reread"]["new_wrong"], 1)  # the partial weight counts as a new wrong read
                self.assertEqual((year["original"]["correct"], year["year_reread"]["blank"]), (1, 1))
                self.assertEqual(year["year_reread"]["correct_reads_lost"], 1)
            with (output / "predictions-tare_reread.csv").open(newline="") as file:
                self.assertEqual(list(csv.reader(file)), [["bottle_group", "tare_kg", "approval_year"],
                                                          ["b1", "10.0", "2034"]])
            with self.assertRaises(SystemExit):
                compare.main(arguments)


if __name__ == "__main__":
    unittest.main()
