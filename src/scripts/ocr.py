"""Read bottle markings locally; optional field boxes say what each value means."""

import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

if __package__:
    from .ocr_selection import load_selection, parse_candidates, parse_value, select_fields
else:
    from ocr_selection import load_selection, parse_candidates, parse_value, select_fields

SCALE = 2             # small stamped characters read better when enlarged,
ENLARGE_BELOW = 1000  # but enlarging regions this large (px) only costs memory
PAD = 4               # pixels kept around a field box so its edge characters are not cut
PROPOSAL_CONFIDENCE = 0.01  # retain weak field proposals for later threshold experiments
YELLOW, GREEN = (255, 220, 0), (0, 200, 0)


def field_candidates(field, readings, indices, box, confidence, source):
    """Score only the tokens used by the value, not a confidently read label or unit."""
    text, spans = "", []
    for index in indices:
        token = readings[index]
        start = len(text)
        text += token["text"] + " "
        spans.append((start, len(text) - 1, index))
    candidates = []
    for parsed in parse_candidates(field, text):
        used = [index for start, end, index in spans if start < parsed["end"] and end > parsed["start"]]
        points = np.concatenate([readings[index]["box"] for index in used])
        candidates.append({"field": field, "value": parsed["value"], "value_text": parsed["value_text"],
                           "text": text.strip(), "box": box,
                           "value_box": [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()],
                           "field_confidence": float(confidence),
                           "ocr_confidence": min(readings[index]["confidence"] for index in used),
                           "reading_indices": used, "source": source, "engine": readings[used[0]]["engine"]})
    return candidates


def enhance(rgb):
    """Grey copy with mild local contrast, enlarged if small; return it and its scale."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    scale = SCALE if max(grey.shape) < ENLARGE_BELOW else 1
    return cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), scale


class BottleOCR:
    """Load the pretrained readers (and optional field detector) once per run.

    With a field detector, PARSeq (docTR) reads each field box as a second engine.
    """

    def __init__(self, device="cpu", fields=None, selection=None):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=False if device == "cpu" else device,
                                     verbose=False)
        self.device = device
        self.selection = selection if selection is not None else load_selection()
        self.fields = None
        if fields:
            from doctr.models import recognition_predictor
            from ultralytics import YOLO

            self.fields = YOLO(str(fields))
            self.second = recognition_predictor("parseq", pretrained=True)

    def text(self, rgb, x0=0, y0=0, source="whole_crop"):
        """Read one region; text boxes are returned in whole-crop coordinates."""
        tokens = []
        image, scale = enhance(rgb)
        for box, text, confidence in self.reader.readtext(image, detail=1):
            points = np.asarray(box, dtype=float) / scale + (x0, y0)
            tokens.append({"text": text, "confidence": float(confidence),
                           "box": points.tolist(), "source": source, "engine": "easyocr"})
        return tokens

    def read(self, image):
        """Accept a PIL bottle crop at its original resolution; return text and fields."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(image)
        height, width = rgb.shape[:2]
        result = {"image_size": [width, height], "readings": self.text(rgb),
                  "proposals": [], "candidates": [], "fields": {}}
        if self.fields is None:
            return result
        centres = [np.mean(token["box"], axis=0) for token in result["readings"]]
        found = self.fields.predict(image, device=self.device, conf=PROPOSAL_CONFIDENCE, verbose=False)[0]
        detections = zip(found.boxes.conf.tolist(), found.boxes.cls.tolist(), found.boxes.xyxy.tolist())
        for confidence, cls, (x1, y1, x2, y2) in sorted(detections, reverse=True):
            name = found.names[int(cls)]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            box = [x1, y1, x2, y2]
            left, top = max(0, x1 - PAD), max(0, y1 - PAD)
            right, bottom = min(width, x2 + PAD), min(height, y2 + PAD)
            tokens = self.text(rgb[top:bottom, left:right], left, top, "field_region")
            # PARSeq reads the same padded box as one line; its read is a separate token.
            text, score = self.second([np.ascontiguousarray(rgb[top:bottom, left:right])])[0]
            tokens.append({"text": text, "confidence": float(score), "source": "field_region", "engine": "parseq",
                           "box": [[left, top], [right, top], [right, bottom], [left, bottom]]})
            indices = list(range(len(result["readings"]), len(result["readings"]) + len(tokens)))
            result["readings"] += tokens
            result["proposals"].append({"field": name, "box": box, "field_confidence": float(confidence),
                                        "reading_indices": indices})
            for engine_indices in (indices[:-1], indices[-1:]):  # EasyOCR tokens, then PARSeq
                result["candidates"] += field_candidates(name, result["readings"], engine_indices, box,
                                                         confidence, "field_region")
            # Whole-crop reads may recover digits missed by the focused pass. The field
            # score is still the proposal's score; text outside every proposal stays raw.
            inside = [index for index, (x, y) in enumerate(centres) if x1 <= x < x2 and y1 <= y < y2]
            result["candidates"] += field_candidates(name, result["readings"], inside, box, confidence, "whole_crop")
        if result["proposals"]:
            result["fields"] = select_fields({"image": result}, self.selection)
        return result


def reread(predictor, rgb, box):
    """Read a detected box with PARSeq and zero extra border."""
    x1, y1, x2, y2 = box
    text, score = predictor([np.ascontiguousarray(rgb[y1:y2, x1:x2])])[0]
    return text, float(score)


def second_reads(predictor, images, views, selection, tare=True, year=True):
    """Targeted rereads for one bottle (#51); the original observations are kept.

    `images` (RGB arrays) and `views` (reader results) share keys, one per camera view.
    Tare: if the bottle tare is withheld, every tare box that passes the field minimum is
    reread and the selection runs again with those reads added. Year: if an accepted year
    has one view and one reader as support, its box is reread and the year is kept only if
    the reread contains it. Any value with that year supports it (2034 and 05/2034 agree).
    The reread sees the same pixels, so it is not another source.
    """
    if any(value is None for limits in selection["fields"].values() for value in limits.values()):
        raise ValueError("Second reads need a selection config with all thresholds set.")
    views, record, start = copy.deepcopy(views), {"tare_reread": False, "year_reread": None}, time.perf_counter()
    fields = select_fields(views, selection)
    tare_minimum = selection["fields"]["tare_kg"]["min_field_confidence"]
    if tare and fields["tare_kg"]["status"] != "accepted":
        for name, view in views.items():
            for proposal in view["proposals"]:
                if proposal["field"] != "tare_kg" or proposal["field_confidence"] < tare_minimum:
                    continue
                record["tare_reread"] = True  # logged only when the reader runs on a box
                (x1, y1, x2, y2), (text, score) = proposal["box"], reread(predictor, images[name], proposal["box"])
                view["readings"].append({"text": text, "confidence": score, "source": "tare_reread",
                                         "engine": "parseq", "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
                view["candidates"] += field_candidates("tare_kg", view["readings"], [len(view["readings"]) - 1],
                                                       proposal["box"], proposal["field_confidence"], "tare_reread")
        fields = select_fields(views, selection)
    approval = fields["approval_until_date"]
    year_minimum = selection["fields"]["approval_until_date"]["min_field_confidence"]
    if year and approval["status"] == "accepted":
        support = [c for alternative in approval["alternatives"] if alternative["value"][:4] == approval["value"][:4]
                   for c in alternative["observations"] if (c["field_confidence"] or 0) >= year_minimum]
        if len({c["view"] for c in support}) == 1 and len({c.get("engine", "easyocr") for c in support}) == 1:
            best = support[0]
            text, _ = reread(predictor, images[best["view"]], best["box"])
            kept = any(c["value"][:4] == approval["value"][:4] for c in parse_candidates("approval_until_date", text))
            record["year_reread"] = {"view": best["view"], "box": best["box"], "text": text, "kept": kept}
            if not kept:
                fields["approval_until_date"] = {**approval, "value": None, "status": "unconfirmed"}
    record["seconds"] = time.perf_counter() - start
    return fields, record


def put_label(canvas, text, x, y, color):
    """Coloured text on a dark strip, readable on any background."""
    x, y = int(x), int(y)
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(canvas, (x, y - height - 3), (x + width + 4, y + 3), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_preview(image, result):
    """Return a copy of the crop with text boxes (yellow) and field proposals (green)."""
    canvas = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    for token in result["readings"]:
        points = np.asarray(token["box"], dtype=np.int32)
        cv2.polylines(canvas, [points], True, YELLOW, 1)
        put_label(canvas, f"{token['text']} {token['confidence']:.2f}",
                  points[0][0], max(12, points[0][1] - 3), YELLOW)
    for field in result["proposals"]:
        x1, y1, x2, y2 = field["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 2)
        put_label(canvas, f"{field['field']} proposal {field['field_confidence']:.2f}",
                  x1, min(canvas.shape[0] - 4, y2 + 14), GREEN)
    return Image.fromarray(canvas)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="Full-resolution bottle crops.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file.")
    parser.add_argument("--fields", type=Path, help="Optional field-box detector weights.")
    parser.add_argument("--selection-config", type=Path, help="Selection config; defaults to ocr_selection.json.")
    parser.add_argument("--same-bottle", action="store_true", help="Explicitly combine these crops as views of one bottle.")
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
    if len({path.resolve() for path in args.images}) != len(args.images):
        raise SystemExit("Select each source image only once.")
    try:
        selection = load_selection(args.selection_config)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    try:
        ocr = BottleOCR(args.device, args.fields, selection)
    except ModuleNotFoundError as error:
        if error.name not in ("easyocr", "ultralytics", "doctr"):
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
        summary = {"engine": "easyocr", "language": "en", "device": args.device,
                   "field_weights": str(args.fields) if args.fields else None,
                   "proposal_confidence": PROPOSAL_CONFIDENCE, "ocr_selection": selection,
                   "same_bottle": args.same_bottle, "images": results}
        if args.same_bottle:
            summary["fields"] = select_fields({r["image_path"]: r for r in results}, selection)
        json.dump(summary, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"Saved {args.output} and previews in {previews}")


if __name__ == "__main__":
    main()
