"""Standalone OCR and OCR-field selection for the modular pipeline."""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from .pipeline_config import (
        OCR_ENLARGE_BELOW, OCR_FIELDS, OCR_GREEN, OCR_MIN_DATE_YEAR,
        OCR_PAD, OCR_PROPOSAL_CONFIDENCE, OCR_SCALE, OCR_YELLOW,
    )
except ImportError:
    from pipeline_config import (
        OCR_ENLARGE_BELOW, OCR_FIELDS, OCR_GREEN, OCR_MIN_DATE_YEAR,
        OCR_PAD, OCR_PROPOSAL_CONFIDENCE, OCR_SCALE, OCR_YELLOW,
    )

NUMERIC = re.compile(r"[0-9]+(?:\s*[.,/-]\s*[0-9]+)*")
NUMBER_EDGES = tuple(".,/+-")
TARE_LABELS = {"t", "tw", "tara", "tarra", "tare"}
WEIGHT_LABEL = re.compile(r"\b(?:tw|tara|tarra|tare|netto|net|butaan|propaan|butane|propane)\b", re.I)
LIMITS = {"min_field_confidence", "min_ocr_confidence", "min_margin"}
CONFIG_PATH = Path(__file__).with_name("ocr_selection.json")


def parse_candidates(field, text):
    candidates = []
    previous_end = 0
    for match in NUMERIC.finditer(text):
        before, after = text[:match.start()], text[match.end():]
        context = text[previous_end:match.start()]
        previous_end = match.end()
        if before.rstrip().endswith(NUMBER_EDGES) or after.lstrip().startswith(NUMBER_EDGES):
            continue
        raw = re.sub(r"\s+", "", match.group())
        value = None
        if field.endswith("_kg") and re.fullmatch(r"\d{1,2}(?:[.,]\d{1,2})?", raw):
            attached_label = re.search(r"[a-z]+$", before, re.I)
            if after[:1].isalpha() and not re.match(r"kg\b", after, re.I):
                continue
            labels = WEIGHT_LABEL.findall(context)
            label = attached_label.group().lower() if attached_label else (labels[-1].lower() if labels else None)
            if label and label not in TARE_LABELS:
                continue
            if re.match(r"\s*(?:l\b|lit(?:er|re))", after, re.I):
                continue
            if not label and re.match(r"\s*(?:kg\s+)?net(?:to)?\b", after, re.I):
                continue
            value = float(raw.replace(",", ".")) or None
        elif field.endswith("_date"):
            if before[-1:].isalnum() or after[:1].isalnum() or re.search(r"\bun\s*$", before, re.I):
                continue
            parts = re.split(r"[/.-]", raw)
            try:
                if len(parts) == 1 and re.fullmatch(r"(?:19|20)\d{2}", raw):
                    value = raw
                elif len(parts) == 2:
                    year, month = parts if len(parts[0]) == 4 else parts[::-1]
                    if len(year) in (2, 4) and 1 <= len(month) <= 2:
                        year = int(year)
                        if year < 100:
                            year += 1900 if year >= 50 else 2000
                        date(year, int(month), 1)
                        if 1900 <= year <= 2099:
                            value = f"{year}-{int(month):02d}"
                elif re.fullmatch(r"(?:19|20)[0-9]{2}-[0-9]{2}-[0-9]{2}", raw):
                    value = date.fromisoformat(raw).isoformat()
            except ValueError:
                pass
            if value is not None and int(value[:4]) < OCR_MIN_DATE_YEAR:
                value = None
        if value is not None:
            candidates.append({"value": value, "start": match.start(), "end": match.end(),
                               "value_text": match.group()})
    return candidates


def parse_value(field, text):
    values = {candidate["value"] for candidate in parse_candidates(field, text)}
    return next(iter(values)) if len(values) == 1 else None


def unit_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def load_selection(path=None):
    config = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"alpha", "fields"} or not unit_score(config["alpha"]):
        raise ValueError("Selection config needs alpha in [0, 1] and fields.")
    if not isinstance(config["fields"], dict) or not config["fields"]:
        raise ValueError("Selection config needs at least one field.")
    for name, limits in config["fields"].items():
        if (not name.endswith(("_kg", "_date")) or not isinstance(limits, dict)
                or set(limits) != LIMITS
                or any(value is not None and not unit_score(value) for value in limits.values())):
            raise ValueError("Invalid selection limits for the OCR field.")
    if set(config["fields"]) != set(OCR_FIELDS):
        raise ValueError("Standalone OCR selection must contain tare_kg and approval_until_date.")
    return config


def select_fields(views, config, provenance=None):
    selected = {}
    for field, limits in config["fields"].items():
        configured = all(value is not None for value in limits.values())
        groups = {}
        for view_name, view in views.items():
            for candidate in view["candidates"]:
                if candidate["field"] != field or candidate["value"] is None:
                    continue
                field_score, text_score = candidate["field_confidence"], candidate["ocr_confidence"]
                if any(score is not None and not unit_score(score) for score in (field_score, text_score)):
                    raise ValueError("Candidate confidence must be finite and in [0, 1].")
                scored = field_score is not None and text_score is not None
                score = field_score ** config["alpha"] * text_score ** (1 - config["alpha"]) if scored else None
                observation = {**candidate, "view": view_name, "score": score}
                if provenance is not None:
                    crop = provenance[view_name]
                    observation.update(camera=view_name, frame=crop["frame"], track_id=crop["track_id"],
                                       bottle_box=crop["box"])
                groups.setdefault(candidate["value"], []).append(observation)
        alternatives = []
        for value, observations in groups.items():
            agreed = len({observation.get("engine", "easyocr") for observation in observations}) > 1
            for observation in observations:
                observation["qualified"] = (configured and observation["score"] is not None
                    and observation["field_confidence"] >= limits["min_field_confidence"]
                    and (agreed or observation["ocr_confidence"] >= limits["min_ocr_confidence"]))
            observations.sort(key=lambda observation: (
                observation["qualified"], observation["score"] if observation["score"] is not None else -1
            ), reverse=True)
            best = observations[0]
            alternatives.append({"value": value, "score": best["score"], "qualified": best["qualified"],
                                 "agreed": agreed,
                                 "support_views": sorted({observation["view"] for observation in observations}),
                                 "observations": observations})
        alternatives.sort(key=lambda candidate: (
            candidate["qualified"], candidate["agreed"],
            candidate["score"] if candidate["score"] is not None else -1
        ), reverse=True)
        eligible = [candidate for candidate in alternatives if candidate["qualified"]]
        rivals = [candidate for candidate in eligible[1:] if candidate["agreed"] == eligible[0]["agreed"]] if eligible else []
        margin = eligible[0]["score"] - rivals[0]["score"] if rivals else None
        if not alternatives:
            status = "missing"
        elif not configured:
            status = "unconfigured"
        elif not any(candidate["score"] is not None for candidate in alternatives):
            status = "unscored"
        elif not eligible:
            status = "low_confidence"
        elif margin is not None and (margin <= 0 or margin < limits["min_margin"]):
            status = "ambiguous"
        else:
            status = "accepted"
        selected[field] = {"value": eligible[0]["value"] if status == "accepted" else None,
                           "status": status, "margin": margin,
                           "best": alternatives[0]["observations"][0] if alternatives else None,
                           "alternatives": alternatives}
    return selected


def field_candidates(field, readings, indices, box, confidence, source):
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
                           "reading_indices": used, "source": source,
                           "engine": readings[used[0]]["engine"]})
    return candidates


def enhance(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    scale = OCR_SCALE if max(grey.shape) < OCR_ENLARGE_BELOW else 1
    return cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), scale


class BottleOCR:
    """Load EasyOCR and optional field/PARSeq readers once per run."""

    def __init__(self, device="cpu", fields=None, selection=None):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=False if device == "cpu" else device, verbose=False)
        self.device = device
        self.selection = selection if selection is not None else load_selection()
        self.fields = None
        if fields:
            from doctr.models import recognition_predictor
            from ultralytics import YOLO
            self.fields = YOLO(str(fields))
            self.second = recognition_predictor("parseq", pretrained=True)

    def text(self, rgb, x0=0, y0=0, source="whole_crop"):
        tokens = []
        image, scale = enhance(rgb)
        for box, text, confidence in self.reader.readtext(image, detail=1):
            points = np.asarray(box, dtype=float) / scale + (x0, y0)
            tokens.append({"text": text, "confidence": float(confidence),
                           "box": points.tolist(), "source": source, "engine": "easyocr"})
        return tokens

    def read(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(image)
        height, width = rgb.shape[:2]
        result = {"image_size": [width, height], "readings": self.text(rgb),
                  "proposals": [], "candidates": [], "fields": {}}
        if self.fields is None:
            return result
        centres = [np.mean(token["box"], axis=0) for token in result["readings"]]
        found = self.fields.predict(image, device=self.device,
                                    conf=OCR_PROPOSAL_CONFIDENCE, verbose=False)[0]
        detections = zip(found.boxes.conf.tolist(), found.boxes.cls.tolist(), found.boxes.xyxy.tolist())
        for confidence, cls, (x1, y1, x2, y2) in sorted(detections, reverse=True):
            name = found.names[int(cls)]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            box = [x1, y1, x2, y2]
            left, top = max(0, x1 - OCR_PAD), max(0, y1 - OCR_PAD)
            right, bottom = min(width, x2 + OCR_PAD), min(height, y2 + OCR_PAD)
            tokens = self.text(rgb[top:bottom, left:right], left, top, "field_region")
            text, score = self.second([np.ascontiguousarray(rgb[top:bottom, left:right])])[0]
            tokens.append({"text": text, "confidence": float(score), "source": "field_region",
                           "engine": "parseq", "box": [[left, top], [right, top],
                           [right, bottom], [left, bottom]]})
            indices = list(range(len(result["readings"]), len(result["readings"]) + len(tokens)))
            result["readings"] += tokens
            result["proposals"].append({"field": name, "box": box, "field_confidence": float(confidence),
                                        "reading_indices": indices})
            for engine_indices in (indices[:-1], indices[-1:]):
                result["candidates"] += field_candidates(name, result["readings"], engine_indices,
                                                          box, confidence, "field_region")
            inside = [index for index, (x, y) in enumerate(centres)
                      if x1 <= x < x2 and y1 <= y < y2]
            result["candidates"] += field_candidates(name, result["readings"], inside,
                                                      box, confidence, "whole_crop")
        if result["proposals"]:
            result["fields"] = select_fields({"image": result}, self.selection)
        return result


def reselect(document, config):
    def update_view(view):
        view["fields"] = select_fields({"image": view}, config) if view["proposals"] else {}

    if "images" in document:
        for result in document["images"]:
            update_view(result)
        if document.get("same_bottle"):
            document["fields"] = select_fields({r["image_path"]: r for r in document["images"]}, config)
    elif "predictions" in document:
        for result in document["predictions"]:
            if "ocr" in result:
                for view in result["ocr"].values():
                    update_view(view)
                result["ocr_fields"] = select_fields(result["ocr"], config, result["crops"])
    else:
        raise ValueError("Expected standalone images or three-camera predictions.")
    document["ocr_selection"] = config
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read tare and approval markings from bottle crops.")
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--selection-config", type=Path)
    parser.add_argument("--same-bottle", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda:0"), default="cpu")
    args = parser.parse_args(argv)
    selection = load_selection(args.selection_config)
    ocr = BottleOCR(args.device, args.fields, selection)
    results = []
    for path in args.images:
        with Image.open(path) as image:
            results.append({"image_path": str(path), **ocr.read(image)})
    document = {"ocr_selection": selection, "same_bottle": args.same_bottle, "images": results}
    if args.same_bottle:
        document["fields"] = select_fields({r["image_path"]: r for r in results}, selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document


__all__ = ["BottleOCR", "load_selection", "select_fields", "parse_candidates", "parse_value"]
