"""Read bottle markings locally; extracted numbers are candidates for review."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def field_candidates(text):
    """Keep explicit kg values and standalone four-digit years, without guessing."""
    weights = re.findall(r"(?<![\w.,])([0-9]{1,3}(?:[.,][0-9]{1,2})?)\s*kg\b", text,
                         flags=re.IGNORECASE)
    years = re.findall(r"(?<!\w)(?<![0-9][.,])((?:19|20)[0-9]{2})(?!\w|[.,][0-9])", text)
    return {
        "weight_kg_candidates": sorted({float(value.replace(",", ".")) for value in weights}),
        "year_candidates": sorted({int(value) for value in years}),
    }


class BottleOCR:
    """Load the pretrained English reader once and reuse it for each crop."""

    def __init__(self, device="cpu"):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=False if device == "cpu" else device,
                                     verbose=False)

    def read(self, image):
        """Accept a PIL bottle crop at its original resolution; return text evidence."""
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        readings = self.reader.readtext(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detail=1)
        tokens = []
        for box, text, confidence in readings:
            tokens.append({
                "text": text,
                "confidence": float(confidence),
                "box": np.asarray(box).astype(float).tolist(),
                **field_candidates(text),
            })
        return {
            "image_size": [rgb.shape[1], rgb.shape[0]],
            "readings": tokens,
            "weight_kg_candidates": sorted({v for t in tokens for v in t["weight_kg_candidates"]}),
            "year_candidates": sorted({v for t in tokens for v in t["year_candidates"]}),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="Full-resolution bottle crops.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file.")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda:0"), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new --output file.")
    for path in args.images:
        if not path.is_file():
            raise SystemExit(f"Image missing: {path}")
        with Image.open(path) as image:
            image.verify()

    try:
        ocr = BottleOCR(args.device)
    except ModuleNotFoundError as error:
        if error.name != "easyocr":
            raise
        raise SystemExit("Install OCR dependencies: python -m pip install -r requirements-ocr.txt") from error
    results = []
    for path in args.images:
        with Image.open(path) as image:
            result = {"image_path": str(path), **ocr.read(image)}
        results.append(result)
        print(f"{path.name}: {len(result['readings'])} text regions", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"engine": "easyocr", "language": "en", "device": args.device,
                   "images": results}, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
