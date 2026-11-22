"""Paired comparison of the targeted second reads (#51) with the original selection.

Uses saved bottle caches (from the team's run_ocr.py) and their source images; only
PARSeq runs again. The detector, readers, parser and selection config stay frozen. Use
development passages that were not used for earlier tuning or testing. Each variant is
scored separately against human markings and workbook references: tare within 0.10 kg,
year by exact match. A tare that only drops its fraction (10 for 10.7) counts as partial.
Each variant's predictions are also written in the team format for score_full_ocr.py.

python src/scripts/compare_second_reads.py <cache> --config <selection.json> --references <references.csv> --output <dir>
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

if __package__:
    from .evaluate_ocr import load_caches
    from .ocr import second_reads
    from .ocr_selection import load_selection, select_fields
else:
    from evaluate_ocr import load_caches
    from ocr import second_reads
    from ocr_selection import load_selection, select_fields

TOLERANCE_KG = 0.10
COLUMNS = {"tare_kg": "tare_kg", "approval_until_date": "approval_year"}  # field -> team column
SOURCES = {"human": "marking", "workbook": "workbook"}  # source -> team column prefix
VARIANTS = {"original": None, "tare_reread": {"tare": True, "year": False},
            "year_reread": {"tare": False, "year": True}}


def load_references(path):
    """Passage references in the team format; a blank cell means unknown."""
    references = {}
    with open(path, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            references[row["bottle_group"]] = {
                (source, field): (float if field == "tare_kg" else int)(cell) if (cell := row[f"{prefix}_{column}"].strip())
                else None for source, prefix in SOURCES.items() for field, column in COLUMNS.items()}
    return references


def outcome(field, accepted, reference):
    """correct, wrong, partial (tare without its fraction) or blank; None when unscored."""
    if reference is None:
        return None
    if accepted is None:
        return "blank"
    if field == "tare_kg":
        if abs(accepted - reference) <= TOLERANCE_KG + 1e-9:
            return "correct"
        return "partial" if accepted == math.floor(reference) else "wrong"
    return "correct" if int(str(accepted)[:4]) == reference else "wrong"


def paired(rows):
    """Counts per variant plus the change against the original on the same bottles."""
    summary = {}
    for source in SOURCES:
        for field in COLUMNS:
            scored = [row for row in rows if row["source"] == source and row["field"] == field
                      and row["outcome"] is not None]
            original = {row["bottle_group"]: row["outcome"] for row in scored if row["variant"] == "original"}
            for variant in VARIANTS:
                mine = [row for row in scored if row["variant"] == variant]
                counts = {kind: sum(row["outcome"] == kind for row in mine)
                          for kind in ("correct", "wrong", "partial", "blank")}
                changes = [(original[row["bottle_group"]], row["outcome"]) for row in mine]
                summary.setdefault(source, {}).setdefault(field, {})[variant] = {
                    "bottles": len(mine), **counts,
                    "correct_reads_lost": sum(old == "correct" and new != "correct" for old, new in changes),
                    "wrong_reads_removed": sum(old in ("wrong", "partial") and new not in ("wrong", "partial")
                                               for old, new in changes),
                    "new_correct": sum(old != "correct" and new == "correct" for old, new in changes),
                    "new_wrong": sum(old not in ("wrong", "partial") and new in ("wrong", "partial")
                                     for old, new in changes)}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", type=Path, help="Folder of bottle caches from run_ocr.py.")
    parser.add_argument("--config", type=Path, required=True, help="Frozen selection config.")
    parser.add_argument("--references", type=Path, required=True, help="Team references CSV (workbook_*, marking_*).")
    parser.add_argument("--output", type=Path, required=True, help="New folder for the comparison.")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new --output folder.")
    config, documents, references = load_selection(args.config), load_caches(args.cache), load_references(args.references)
    if not documents or set(documents) - set(references):
        raise SystemExit("Every bottle cache needs a row in the references file.")
    from doctr.models import recognition_predictor

    predictor = recognition_predictor("parseq", pretrained=True)
    rows, predictions, seconds = [], {variant: [] for variant in VARIANTS}, {variant: [] for variant in VARIANTS}
    for group, document in sorted(documents.items()):
        views, images = {view["image_path"]: view for view in document["images"]}, {}
        for path in views:
            with Image.open(path) as image:
                images[path] = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        for variant, flags in VARIANTS.items():
            if flags is None:
                fields, record = select_fields(views, config), {"seconds": 0.0}
            else:
                fields, record = second_reads(predictor, images, views, config, **flags)
            seconds[variant].append(record["seconds"])
            accepted = {field: fields[field]["value"] if fields[field]["status"] == "accepted" else None
                        for field in COLUMNS}
            predictions[variant].append([group, "" if accepted["tare_kg"] is None else accepted["tare_kg"],
                                         "" if accepted["approval_until_date"] is None
                                         else str(accepted["approval_until_date"])[:4]])
            for field in COLUMNS:
                for source in SOURCES:
                    reference = references[group][(source, field)]
                    rows.append({"bottle_group": group, "variant": variant, "field": field, "source": source,
                                 "reference": reference, "accepted": accepted[field], "status": fields[field]["status"],
                                 "outcome": outcome(field, accepted[field], reference),
                                 "tare_reread": record.get("tare_reread"), "year_reread": record.get("year_reread")})

    args.output.mkdir(parents=True)
    for variant, lines in predictions.items():
        with open(args.output / f"predictions-{variant}.csv", "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows([["bottle_group", "tare_kg", "approval_year"], *lines])
    with open(args.output / "bottles.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"bottles": len(documents), "tare_tolerance_kg": TOLERANCE_KG, "selection": config,
              "extra_seconds_per_bottle": {variant: sum(values) / len(values) for variant, values in seconds.items()},
              "comparison": paired(rows)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("bottles", "extra_seconds_per_bottle", "comparison")}, indent=2))
    print(f"Saved the comparison in {args.output}")


if __name__ == "__main__":
    main()
