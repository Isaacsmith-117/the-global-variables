import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from src.scripts import ocr


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


def bottle_ocr(text, fields=None):
    """Build BottleOCR with fake EasyOCR/YOLO modules, so the tests need neither."""
    easyocr, ultralytics = Mock(), Mock()
    easyocr.Reader.return_value = FakeReader(text)
    ultralytics.YOLO.return_value = fields
    with patch.dict(sys.modules, {"easyocr": easyocr, "ultralytics": ultralytics}):
        return ocr.BottleOCR("cpu", "fields.pt" if fields else None)


class OCRTests(unittest.TestCase):
    def test_field_class_suffix_selects_the_parser(self):
        cases = [("tare_kg", "TW 13,4 kg", 13.4), ("tare_kg", "T13.4", 13.4), ("tare_kg", "11", 11.0),
                 ("tare_kg", "TW", None), ("tare_kg", "113.4", None),
                 ("test_date", "03/25", "2025-03"), ("test_date", "3-2025", "2025-03"),
                 ("manufacture_date", "2015/03", "2015-03"), ("manufacture_date", "12.2019", "2019-12"),
                 ("manufacture_date", "05/98", "1998-05"), ("test_date", "2030", "2030"),
                 ("test_date", "13/25", None), ("test_date", "kg", None), ("serial", "2025", None)]
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

    def test_field_boxes_keep_the_best_box_and_map_text_back_to_the_crop(self):
        model = field_model([("test_date", 0.4, [0, 0, 10, 10]), ("test_date", 0.8, [20, 10, 50, 30]),
                             ("tare_kg", 0.7, [56, 0, 60, 8])])
        reader = bottle_ocr("03/25", model)
        result = reader.read(Image.new("RGB", (60, 40), "white"))
        self.assertEqual(set(result["fields"]), {"test_date", "tare_kg"})
        date = result["fields"]["test_date"]
        self.assertEqual(date["box"], [16, 6, 54, 34])  # padded by 4 px, kept inside the crop
        self.assertEqual(date["box_confidence"], 0.8)
        self.assertEqual(date["value"], "2025-03")
        self.assertEqual(result["fields"]["tare_kg"]["box"], [52, 0, 60, 12])
        self.assertIn([17.0, 7.0], [token["box"][0] for token in result["readings"]])

    def test_preview_draws_boxes_on_a_copy(self):
        image = Image.new("RGB", (60, 40), "white")
        result = bottle_ocr("03/25", field_model([("test_date", 0.8, [20, 10, 50, 30])])).read(image)
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
            self.assertEqual(summary["images"][0]["readings"][0]["text"], "TW 13,4")
            self.assertTrue(Path(summary["images"][0]["preview"]).is_file())


if __name__ == "__main__":
    unittest.main()
