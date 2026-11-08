"""Read bottle markings locally; optional field boxes say what each value means."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SCALE = 2  # small stamped characters read better when enlarged
PAD = 4    # pixels kept around a field box so its edge characters are not cut
NUMBER = re.compile(r"(?<![0-9.,])([0-9]{1,2}(?:[.,][0-9]{1,2})?)(?![0-9])")
# YYYY, YYYY/MM, MM/YY or MM/YYYY, separated by / . or -
DATE = re.compile(r"(?<![0-9])(?:((?:19|20)[0-9]{2})(?:\s*[/.-]\s*([0-9]{1,2}))?"
                  r"|([0-9]{1,2})\s*[/.-]\s*((?:19|20)?[0-9]{2}))(?![0-9])")
YELLOW, GREEN = (255, 220, 0), (0, 200, 0)


def parse_value(field, text):
    """Field classes end in _kg or _date; the box class gives the meaning, the text the value."""
    if field.endswith("_kg"):
        match = NUMBER.search(text)
        return float(match.group(1).replace(",", ".")) if match else None
    if field.endswith("_date"):
        match = DATE.search(text)
        if not match:
            return None
        year, month_after, month, short_year = match.groups()
        year = int(year or short_year)
        if year < 100:
            year += 1900 if year >= 50 else 2000
        month = month or month_after
        if month is None:
            return str(year)
        return f"{year}-{int(month):02d}" if 1 <= int(month) <= 12 else None
    return None


def enhance(rgb):
    """Grey image with mild local contrast, enlarged; the original crop is not changed."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    return cv2.resize(grey, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)


class BottleOCR:
    """Load the pretrained English reader (and optional field detector) once per run."""

    def __init__(self, device="cpu", fields=None):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=False if device == "cpu" else device,
                                     verbose=False)
        self.device = device
        self.fields = None
        if fields:
            from ultralytics import YOLO

            self.fields = YOLO(str(fields))

    def text(self, rgb, x0=0, y0=0):
        """Read one region; text boxes are returned in whole-crop coordinates."""
        tokens = []
        for box, text, confidence in self.reader.readtext(enhance(rgb), detail=1):
            points = np.asarray(box, dtype=float) / SCALE + (x0, y0)
            tokens.append({"text": text, "confidence": float(confidence), "box": points.tolist()})
        return tokens

    def read(self, image):
        """Accept a PIL bottle crop at its original resolution; return text and fields."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(image)
        height, width = rgb.shape[:2]
        result = {"image_size": [width, height], "readings": [], "fields": {}}
        if self.fields is None:
            result["readings"] = self.text(rgb)
            return result
        found = self.fields.predict(image, device=self.device, verbose=False)[0]
        detections = zip(found.boxes.conf.tolist(), found.boxes.cls.tolist(), found.boxes.xyxy.tolist())
        for confidence, cls, (x1, y1, x2, y2) in sorted(detections, reverse=True):
            name = found.names[int(cls)]
            if name in result["fields"]:
                continue  # keep only the most confident box of each field
            x1, y1 = max(0, int(x1) - PAD), max(0, int(y1) - PAD)
            x2, y2 = min(width, int(x2) + PAD), min(height, int(y2) + PAD)
            tokens = self.text(rgb[y1:y2, x1:x2], x1, y1)
            text = " ".join(token["text"] for token in tokens)
            result["readings"] += tokens
            result["fields"][name] = {"value": parse_value(name, text), "text": text,
                                      "box": [x1, y1, x2, y2], "box_confidence": confidence}
        return result


def put_label(canvas, text, x, y, color):
    """Coloured text on a dark strip, readable on any background."""
    x, y = int(x), int(y)
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(canvas, (x, y - height - 3), (x + width + 4, y + 3), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_preview(image, result):
    """Return a copy of the crop with text boxes (yellow) and parsed field boxes (green)."""
    canvas = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    for token in result["readings"]:
        points = np.asarray(token["box"], dtype=np.int32)
        cv2.polylines(canvas, [points], True, YELLOW, 1)
        put_label(canvas, f"{token['text']} {token['confidence']:.2f}",
                  points[0][0], max(12, points[0][1] - 3), YELLOW)
    for name, field in result["fields"].items():
        x1, y1, x2, y2 = field["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 2)
        value = "?" if field["value"] is None else field["value"]
        put_label(canvas, f"{name}: {value}", x1, min(canvas.shape[0] - 4, y2 + 14), GREEN)
    return Image.fromarray(canvas)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="Full-resolution bottle crops.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file.")
    parser.add_argument("--fields", type=Path, help="Optional field-box detector weights.")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda:0"), default="cpu")
    args = parser.parse_args(argv)
    previews = args.output.with_name(f"{args.output.stem}_previews")
    if args.output.exists() or previews.exists():
        raise SystemExit("Output already exists; choose a new --output file.")
    for path in [*args.images, *([args.fields] if args.fields else [])]:
        if not path.is_file():
            raise SystemExit(f"File missing: {path}")
    for path in args.images:
        with Image.open(path) as image:
            image.verify()

    try:
        ocr = BottleOCR(args.device, args.fields)
    except ModuleNotFoundError as error:
        if error.name not in ("easyocr", "ultralytics"):
            raise
        raise SystemExit("Install OCR dependencies: python -m pip install -r requirements-ocr.txt "
                         "(and requirements.txt for --fields)") from error
    previews.mkdir(parents=True)
    results = []
    for n, path in enumerate(args.images, start=1):
        preview = previews / f"{n:03d}_{path.stem}.jpg"
        with Image.open(path) as image:
            result = {"image_path": str(path), "preview": str(preview), **ocr.read(image)}
            draw_preview(image, result).save(preview)
        results.append(result)
        print(f"{path.name}: {len(result['readings'])} text regions, "
              f"{len(result['fields'])} fields", flush=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"engine": "easyocr", "language": "en", "device": args.device,
                   "field_weights": str(args.fields) if args.fields else None,
                   "images": results}, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Saved {args.output} and previews in {previews}")


if __name__ == "__main__":
    main()
