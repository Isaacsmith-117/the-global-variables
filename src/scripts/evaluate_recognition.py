"""Measure EasyOCR on hand-drawn field boxes against the human transcriptions.

Separates recognition from detection: each labelled box with literal text is read with
a few reader variants and compared as exact text, character error rate and parsed value.
Use development splits only (train/validation); the test split stays untouched.

python src/scripts/evaluate_recognition.py data/ocr_v2 --output src/runs/ocr-fields/recognition
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

if __package__:
    from .ocr import PAD, enhance
    from .ocr_selection import parse_value
else:
    from ocr import PAD, enhance
    from ocr_selection import parse_value

ALLOWLIST = {"tare_kg": "0123456789.,KGTAREW", "approval_until_date": "0123456789/.-"}
VARIANTS = ("current", "allowlist", "deskew", "deskew_allowlist")


def corners(value, width, height):
    """Label Studio rectangle (percent, rotated clockwise about its top-left) in pixels."""
    x, y = value["x"] * width / 100, value["y"] * height / 100
    w, h = value["width"] * width / 100, value["height"] * height / 100
    angle = math.radians(value.get("rotation", 0))
    right, down = np.array([math.cos(angle), math.sin(angle)]), np.array([-math.sin(angle), math.cos(angle)])
    origin = np.array([x, y])
    return np.array([origin, origin + w * right, origin + w * right + h * down, origin + h * down])


def region(rgb, value, deskew):
    """Crop the enclosing box, or first rotate the image so the box is axis-aligned."""
    height, width = rgb.shape[:2]
    if deskew and value.get("rotation", 0):
        origin = (value["x"] * width / 100, value["y"] * height / 100)
        matrix = cv2.getRotationMatrix2D(origin, value["rotation"], 1.0)
        rgb = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        value = {**value, "rotation": 0}
    points = corners(value, width, height)
    x1, y1 = np.floor(points.min(axis=0)).astype(int) - PAD
    x2, y2 = np.ceil(points.max(axis=0)).astype(int) + PAD
    return rgb[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def normal(text):
    return "".join(text.upper().split())


def regions(packet, splits):
    """Yield (row, rectangle value) for labelled boxes with literal text."""
    rectangles = {}
    for split in ("train", "val"):
        for task in json.loads((packet / "annotations" / f"{split}.native.json").read_text(encoding="utf-8")):
            for annotation in task["annotations"]:
                for result in annotation["result"]:
                    if result["type"] == "rectanglelabels":
                        rectangles[result["id"]] = result["value"]
    with open(packet / "annotations" / "regions_and_transcriptions.csv", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] in splits and row["literal_text"].strip() and row["region_id"] in rectangles:
                yield row, rectangles[row["region_id"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packet", type=Path, help="Extracted dataset packet, e.g. data/ocr_v2.")
    parser.add_argument("--output", type=Path, required=True, help="New folder for the results.")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    parser.add_argument("--device", choices=("cpu", "mps", "cuda:0"), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new --output folder.")
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False if args.device == "cpu" else args.device, verbose=False)
    args.output.mkdir(parents=True)
    rows = []
    for row, value in regions(args.packet, set(args.splits)):
        with Image.open(args.packet / row["image_path"]) as image:
            rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        truth = row["literal_text"]
        for variant in VARIANTS:
            crop = region(rgb, value, deskew=variant.startswith("deskew"))
            grey, _ = enhance(crop)
            allowlist = ALLOWLIST[row["field"]] if variant.endswith("allowlist") else None
            readings = reader.readtext(grey, detail=1, allowlist=allowlist)
            text = " ".join(reading[1] for reading in readings)
            expected = parse_value(row["field"], truth)
            rows.append({"split": row["split"], "bottle_group": row["bottle_group"], "region_id": row["region_id"],
                         "field": row["field"], "variant": variant, "rotation": value.get("rotation", 0),
                         "truth": truth, "read": text,
                         "confidence": min((float(reading[2]) for reading in readings), default=None),
                         "exact": normal(text) == normal(truth),
                         "cer": edit_distance(normal(text), normal(truth)) / max(1, len(normal(truth))),
                         "truth_value": expected,
                         "value_match": None if expected is None else parse_value(row["field"], text) == expected})
        print(f"{len(rows) // len(VARIANTS)} boxes read", flush=True)

    with open(args.output / "regions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for field in ALLOWLIST:
        for variant in VARIANTS:
            group = [r for r in rows if r["field"] == field and r["variant"] == variant]
            parsed = [r for r in group if r["value_match"] is not None]
            summary.setdefault(field, {})[variant] = {
                "boxes": len(group), "exact_text": sum(r["exact"] for r in group) / len(group),
                "mean_cer": sum(r["cer"] for r in group) / len(group),
                "boxes_with_parseable_truth": len(parsed),
                "value_match": sum(r["value_match"] for r in parsed) / len(parsed) if parsed else None}
    (args.output / "metrics.json").write_text(json.dumps({"splits": args.splits, "metrics": summary}, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (field, by_variant) in zip(axes, summary.items()):
        ax.bar(VARIANTS, [m["value_match"] or 0 for m in by_variant.values()], color="#2e7d32", label="parsed value correct")
        ax.plot(VARIANTS, [m["exact_text"] for m in by_variant.values()], "ko-", label="exact text")
        ax.set(ylim=(0, 1), title=f"{field} ({next(iter(by_variant.values()))['boxes']} boxes)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("share of hand-drawn boxes")
    fig.tight_layout()
    fig.savefig(args.output / "recognition.png", dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
