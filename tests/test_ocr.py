import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from src.scripts import ocr
from src.scripts.ocr_selection import load_selection, select_fields


class FakeReader:
    """Return one text box, in the coordinates of the enlarged region it receives."""

    def __init__(self, text):
        self.text = text
        self.shapes = []

    def readtext(self, image, detail=1):
        self.shapes.append(image.shape)
        return [([[2, 2], [20, 2], [20, 10], [2, 10]], self.text, 0.9)]


def field_model(detections):
    """A YOLO-like model returning (name, confidence, xyxy) detections."""
    names = dict(enumerate(sorted({name for name, _, _ in detections})))
    ids = {name: i for i, name in names.items()}
    found = Mock(names=names)
    found.boxes.conf = np.array([conf for _, conf, _ in detections])
    found.boxes.cls = np.array([ids[name] for name, _, _ in detections], dtype=float)
    found.boxes.xyxy = np.array([box for _, _, box in detections], dtype=float)
    model = Mock()
    model.predict.return_value = [found]
    return model


def bottle_ocr(text, fields=None, selection=None, second=("", 0.0)):
    """Build BottleOCR with fake EasyOCR/YOLO/PARSeq modules, so the tests need none of them."""
    easyocr, ultralytics, doctr = Mock(), Mock(), Mock()
    easyocr.Reader.return_value = FakeReader(text)
    ultralytics.YOLO.return_value = fields
    doctr.models.recognition_predictor.return_value = Mock(side_effect=lambda crops: [second] * len(crops))
    modules = {"easyocr": easyocr, "ultralytics": ultralytics, "doctr": doctr, "doctr.models": doctr.models}
    with patch.dict(sys.modules, modules):
        return ocr.BottleOCR("cpu", "fields.pt" if fields else None, selection)


def configured_selection():
    config = load_selection()
    for limits in config["fields"].values():
        limits.update(min_field_confidence=0.5, min_ocr_confidence=0.5, min_margin=0.05)
    return config


def candidate(value, field_score, text_score):
    return {"field": "tare_kg", "value": value, "field_confidence": field_score,
            "ocr_confidence": text_score, "box": [8, 4, 40, 30], "source": "field_region"}


class OCRTests(unittest.TestCase):
    def test_field_class_suffix_selects_the_parser(self):
        cases = [("tare_kg", "TW 13,4 kg", 13.4), ("tare_kg", "T13.4", 13.4), ("tare_kg", "11", 11.0),
                 ("tare_kg", "TW", None), ("tare_kg", "113.4", None),
                 ("test_date", "03/25", "2025-03"), ("test_date", "3-2025", "2025-03"),
                 ("manufacture_date", "2015/03", "2015-03"), ("manufacture_date", "12.2019", "2019-12"),
                 ("manufacture_date", "05/98", None), ("test_date", "2030", "2030"),
                 ("test_date", "1999", None), ("test_date", "12/1999", None),
                 ("test_date", "1999-12-31", None), ("test_date", "2000", "2000"),
                 ("test_date", "01/00", "2000-01"), ("test_date", "2000-01-01", "2000-01-01"),
                 ("test_date", "1999 2034", "2034"),
                 ("test_date", "13/25", None), ("test_date", "kg", None), ("serial", "2025", None),
                 ("tare_kg", "13..4 kg", None), ("tare_kg", "13.456 kg", None),
                 ("tare_kg", "26.2 L", None), ("tare_kg", "A13.4", None),
                 ("tare_kg", "13 KG NETTO", None), ("tare_kg", "13.4 10.7", None),
                 ("tare_kg", "TARRA 10.7 KG NETTO 13 KG", 10.7),
                 ("tare_kg", "PROPANE 13 KG TARRA 10.7", 10.7),
                 ("test_date", "03/04/2025", None), ("test_date", "UN1965 2034", "2034"),
                 ("test_date", "UN 1965 2034", "2034"),
                 ("test_date", "2034 2035", None), ("test_date", "2032-02-29", "2032-02-29"),
                 ("test_date", "2033-02-29", None), ("test_date", "2034-003-001", None)]
        for field, text, value in cases:
            with self.subTest(field=field, text=text):
                self.assertEqual(ocr.parse_value(field, text), value)

    def test_without_field_model_reads_whole_crop_and_guesses_no_fields(self):
        reader = bottle_ocr("TW 13,4")
        result = reader.read(Image.new("RGB", (40, 30), "white"))
        self.assertEqual(reader.reader.shapes, [(60, 80)])  # grey and enlarged 2x
        self.assertEqual(result["image_size"], [40, 30])
        self.assertEqual(result["readings"][0]["box"][0], [1.0, 1.0])
        self.assertEqual(result["fields"], {})

    def test_large_crops_are_not_enlarged(self):
        reader = bottle_ocr("TW 13,4")
        result = reader.read(Image.new("RGB", (1200, 800), "white"))
        self.assertEqual(reader.reader.shapes, [(800, 1200)])
        self.assertEqual(result["readings"][0]["box"][0], [2.0, 2.0])

    def test_alternative_field_boxes_and_whole_crop_recovery(self):
        field = "approval_until_date"
        model = field_model([(field, 0.98, [10, 10, 40, 30]), (field, 0.8, [50, 40, 80, 65]),
                             ("tare_kg", 0.88, [85, 5, 99, 24])])
        reader = bottle_ocr("", model, configured_selection())
        small = [[2, 2], [20, 2], [20, 10], [2, 10]]
        whole_date = [[110, 90], [140, 90], [140, 110], [110, 110]]
        reader.reader.readtext = Mock(side_effect=[
            [(small, "UN1965", 0.99), (whole_date, "2034", 0.95)],
            [(small, "2035", 0.15)],
            [(small, word, score) for word, score in
             [("TARRA", 0.99), ("10", 0.7), (",", 0.95), ("7", 0.85), ("kg", 0.99)]],
            [],
        ])
        result = reader.read(Image.new("RGB", (100, 80), "white"))
        self.assertEqual(len(result["proposals"]), 3)
        self.assertEqual(result["readings"][0]["text"], "UN1965")
        self.assertEqual(result["readings"][2]["box"][0], [7.0, 7.0])
        self.assertEqual({c["value"] for c in result["candidates"]}, {10.7, "2034", "2035"})
        chosen = result["fields"][field]
        self.assertEqual(chosen["value"], "2034")
        self.assertEqual(chosen["best"]["box"], [50, 40, 80, 65])
        self.assertEqual(chosen["best"]["source"], "whole_crop")
        self.assertEqual(chosen["best"]["field_confidence"], 0.8)
        self.assertEqual(result["fields"]["tare_kg"]["best"]["ocr_confidence"], 0.7)

    def test_ranking_uses_both_scores_and_distinct_values(self):
        views = {"front": {"candidates": [candidate(10.7, 0.95, 0.5), candidate(10.7, 0.95, 0.5)]},
                 "left": {"candidates": [candidate(10.7, 0.9, 0.5)]},
                 "right": {"candidates": [candidate(13.4, 0.72, 0.95)]}}
        config = configured_selection()
        result = select_fields(views, config)["tare_kg"]
        self.assertEqual(result["value"], 13.4)
        self.assertEqual(result["alternatives"][1]["support_views"], ["front", "left"])
        self.assertEqual(len(result["alternatives"]), 2)
        config["fields"]["tare_kg"]["min_margin"] = 0.15
        result = select_fields(views, config)["tare_kg"]
        self.assertIsNone(result["value"])
        self.assertEqual(result["status"], "ambiguous")
        config["alpha"] = 0.9
        config["fields"]["tare_kg"]["min_margin"] = 0.05
        self.assertEqual(select_fields(views, config)["tare_kg"]["value"], 10.7)

    def test_values_read_by_both_engines_are_agreed(self):
        config = configured_selection()
        misread = candidate(18.7, 0.9, 1.0)  # EasyOCR, confidently wrong
        easy, parseq = candidate(10.7, 0.9, 0.3), {**candidate(10.7, 0.9, 0.4), "engine": "parseq"}
        result = select_fields({"front": {"candidates": [misread, easy]},
                                "left": {"candidates": [parseq]}}, config)["tare_kg"]
        self.assertEqual((result["value"], result["status"]), (10.7, "accepted"))
        self.assertTrue(result["alternatives"][0]["agreed"])
        self.assertIsNone(result["margin"])  # no other agreed value competes
        # With one engine only, the confident misread still wins, as before.
        self.assertEqual(select_fields({"front": {"candidates": [misread, easy]}}, config)["tare_kg"]["value"], 18.7)
        weak_box = {**candidate(10.7, 0.4, 0.9), "engine": "parseq"}
        result = select_fields({"front": {"candidates": [candidate(10.7, 0.4, 0.9), weak_box]}}, config)["tare_kg"]
        self.assertEqual(result["status"], "low_confidence")  # agreement does not skip the field minimum

    def test_parseq_reads_each_field_box_as_a_second_engine(self):
        model = field_model([("tare_kg", 0.9, [10, 10, 40, 30])])
        reader = bottle_ocr("TARE 10.7 KG", model, configured_selection(), second=("TARE 10.7KG", 0.4))
        result = reader.read(Image.new("RGB", (60, 40), "white"))
        parseq = [reading for reading in result["readings"] if reading["engine"] == "parseq"]
        self.assertEqual(parseq[0]["box"], [[6, 6], [44, 6], [44, 34], [6, 34]])
        self.assertEqual(reader.second.call_args.args[0][0].shape, (28, 38, 3))
        self.assertEqual({c["engine"] for c in result["candidates"]}, {"easyocr", "parseq"})
        chosen = result["fields"]["tare_kg"]
        self.assertEqual(chosen["value"], 10.7)
        self.assertTrue(chosen["alternatives"][0]["agreed"])

    def test_missing_weak_unscored_and_tied_reads_stay_unknown(self):
        config = configured_selection()
        config["fields"]["tare_kg"]["min_margin"] = 0
        cases = [([], "missing"), ([candidate(10.7, None, 0.99)], "unscored"),
                 ([candidate(10.7, 0.49, 0.99), candidate(13.4, 0.99, 0.49)], "low_confidence"),
                 ([candidate(10.7, 0.8, 0.8), candidate(13.4, 0.8, 0.8)], "ambiguous")]
        for observations, status in cases:
            with self.subTest(status=status):
                result = select_fields({"front": {"candidates": observations}}, config)["tare_kg"]
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["value"])
        views = {"front": {"candidates": [candidate(10.7, 0.9, 0.9)]}}
        result = select_fields(views, load_selection())["tare_kg"]
        self.assertEqual(result["status"], "unconfigured")
        self.assertIsNone(result["value"])
        self.assertEqual(result["best"]["value"], 10.7)
        self.assertEqual(select_fields(views, config)["tare_kg"]["value"], 10.7)

    def test_cache_replay_updates_selections_without_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cached.json"
            output = root / "reselected.json"
            config = root / "selection.json"
            config.write_text(json.dumps(configured_selection()))
            model = field_model([("tare_kg", 0.9, [10, 10, 40, 30])])
            reading = bottle_ocr("10.7", model).read(Image.new("RGB", (60, 40)))
            self.assertIsNone(reading["fields"]["tare_kg"]["value"])
            document = {"predictions": [{"status": "NOK", "ocr": {"front": reading},
                                         "crops": {"front": {"frame": 12, "track_id": 3,
                                                             "box": [20, 30, 80, 70]}}}]}
            source.write_text(json.dumps(document))
            original = source.read_bytes()
            command = [sys.executable, "-S", str(Path(ocr.__file__).with_name("ocr_selection.py")),
                       str(source), "--config", str(config), "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True)
            prediction = json.loads(output.read_text())["predictions"][0]
            self.assertEqual(prediction["status"], "NOK")
            self.assertEqual(prediction["ocr"]["front"]["readings"], reading["readings"])
            self.assertEqual(prediction["ocr"]["front"]["fields"]["tare_kg"]["value"], 10.7)
            winner = prediction["ocr_fields"]["tare_kg"]["best"]
            self.assertEqual(winner["box"], [10, 10, 40, 30])
            self.assertEqual(winner["bottle_box"], [20, 30, 80, 70])
            self.assertEqual((winner["camera"], winner["frame"], winner["track_id"]), ("front", 12, 3))
            self.assertEqual(source.read_bytes(), original)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_preview_draws_boxes_on_a_copy(self):
        image = Image.new("RGB", (60, 40), "white")
        result = bottle_ocr("03/25", field_model([("approval_until_date", 0.8, [20, 10, 50, 30])])).read(image)
        preview = np.array(ocr.draw_preview(image, result))
        self.assertEqual(preview.shape, (40, 60, 3))
        self.assertTrue((preview == ocr.GREEN).all(axis=2).any())
        self.assertTrue((np.array(image) == 255).all())

    def test_command_saves_json_and_previews_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop = root / "b001_front.jpg"
            Image.new("RGB", (40, 30), "white").save(crop)
            output = root / "ocr" / "b001.json"
            with patch.object(ocr, "BottleOCR", return_value=bottle_ocr("TW 13,4")):
                ocr.main([str(crop), "--output", str(output)])
                with self.assertRaises(SystemExit):
                    ocr.main([str(crop), "--output", str(output)])
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(summary["field_weights"])
            self.assertNotIn("fields", summary)
            self.assertEqual(summary["images"][0]["readings"][0]["text"], "TW 13,4")
            self.assertTrue(Path(summary["images"][0]["preview"]).is_file())


if __name__ == "__main__":
    unittest.main()
