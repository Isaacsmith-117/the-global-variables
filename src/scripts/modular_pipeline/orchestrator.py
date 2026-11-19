"""Orchestrate detection, OCR, visual classification and final bottle decisions."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

try:
    from .m1_video_parser import VideoParser
    from .m3_ocr import BottleOCR, load_selection, select_fields
    from .m2_classifier import BottleClassifier
    from .pipeline_config import (
        DEFAULT_VIDEOS, DEVICE_CHOICES, MAX_CROSSING_SPREAD_FRAMES,
        PROJECT_ROOT, RUNS_DIR, VIDEO_CAMERAS,
    )
except ImportError:
    # Direct execution: python src/scripts/modular_pipeline/orchestrator.py
    local_dir = str(Path(__file__).resolve().parent)
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    from m1_video_parser import VideoParser
    from m3_ocr import BottleOCR, load_selection, select_fields
    from m2_classifier import BottleClassifier
    from pipeline_config import (
        DEFAULT_VIDEOS, DEVICE_CHOICES, MAX_CROSSING_SPREAD_FRAMES,
        PROJECT_ROOT, RUNS_DIR, VIDEO_CAMERAS,
    )

import torch


def select_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


DEFAULT_VIDEOS = [
    PROJECT_ROOT / "data" / "primagaz_vids" / "14_55" / f"14_55_{name}_cropped.mp4"
    for name in VIDEO_CAMERAS
]


def inspect_crossings(classifier, captures, tracks, events, offsets, start, end, ocr=None):
    """Build one full-resolution three-view crop triplet and inspect it once."""
    if len({len(camera_events) for camera_events in events}) != 1:
        raise ValueError("Camera counts differ; check bottle matching before inspection.")
    by_track = [dict(tuple(frame_tracks.groupby("track_id"))) for frame_tracks in tracks]
    predictions = {}

    for index in range(len(events[0])):
        groups = [camera_groups[events[camera_index][index][0]]
                  for camera_index, camera_groups in enumerate(by_track)]
        if max(int(group.frame.max()) - offset for group, offset in zip(groups, offsets)) < start:
            continue
        if min(int(group.frame.min()) - offset for group, offset in zip(groups, offsets)) >= end:
            continue

        bottle_id = index + 1
        ready_frame = max(events[camera_index][index][1] - offsets[camera_index]
                          for camera_index in range(3))
        result = {"bottle_id": bottle_id, "status": "INCOMPLETE", "reasons": [],
                  "ready_frame": ready_frame}

        aligned = [events[camera_index][index][1] - offsets[camera_index]
                   for camera_index in range(3)]
        if max(aligned) - min(aligned) > MAX_CROSSING_SPREAD_FRAMES:
            predictions[bottle_id] = {**result, "error": "Camera crossing times do not match."}
            continue

        crops = {}
        boxes = {}
        for camera_index, (camera, capture, rows) in enumerate(zip(VIDEO_CAMERAS, captures, groups)):
            frame_index = events[camera_index][index][1]
            box_row = rows[rows.frame == frame_index]
            if box_row.empty:
                break
            box_row = box_row.iloc[0]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            readable, frame = capture.read()
            if not readable:
                break
            height, width = frame.shape[:2]
            x1, y1 = max(0, int(box_row.x1)), max(0, int(box_row.y1))
            x2, y2 = min(width, int(box_row.x2)), min(height, int(box_row.y2))
            if x2 <= x1 or y2 <= y1:
                break
            crops[camera] = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            boxes[camera] = {"frame": frame_index, "track_id": int(box_row.track_id),
                             "box": [x1, y1, x2, y2]}

        if len(crops) == 3:
            if classifier is not None:
                result.update(classifier.predict(crops, bottle_id))
            else:
                result["status"] = "NOT_CHECKED"
            result["crops"] = boxes
            if ocr is not None:
                result["ocr"] = {camera: ocr.read(image) for camera, image in crops.items()}
                result["ocr_fields"] = select_fields(result["ocr"], ocr.selection, boxes)
        else:
            result["error"] = "Could not read three usable crops."
        predictions[bottle_id] = result

    return predictions


def _approval_year(value):
    if value is None:
        return None
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None


def make_final_decision(prediction, current_year=None):
    """Minimal final policy: visual defects or expired approval => NOK; uncertainty => INCOMPLETE."""
    current_year = current_year or date.today().year
    visual_status = prediction.get("status", "INCOMPLETE")
    ocr_fields = prediction.get("ocr_fields", {})
    tare = ocr_fields.get("tare_kg", {})
    approval = ocr_fields.get("approval_until_date", {})

    tare_value = tare.get("value") if tare.get("status") == "accepted" else None
    approval_year = _approval_year(approval.get("value")) if approval.get("status") == "accepted" else None
    approval_valid = approval_year >= current_year if approval_year is not None else None

    if visual_status == "NOK":
        final_status = "NOK"
    elif visual_status == "INCOMPLETE":
        final_status = "INCOMPLETE"
    elif approval_valid is False:
        final_status = "NOK"
    elif tare.get("status") != "accepted" or approval.get("status") != "accepted":
        final_status = "INCOMPLETE"
    elif visual_status == "OK" and tare_value is not None and approval_valid is True:
        final_status = "OK"
    else:
        final_status = "INCOMPLETE"

    return {
        "final_status": final_status,
        "tare_kg": tare_value,
        "approval_until_year": approval_year,
        "approval_valid": approval_valid,
        "tare_status": tare.get("status", "not_checked"),
        "approval_status": approval.get("status", "not_checked"),
    }


def inspect_counts(predictions, frame, start):
    completed = [prediction for prediction in predictions.values()
                 if start <= prediction["ready_frame"] <= frame]
    return {
        status: sum(prediction["status"] == status for prediction in completed)
        for status in ("OK", "NOK", "INCOMPLETE")
    }


def export_bottle_csv(output, predictions):
    """Write one flattened row per bottle."""
    rows = []
    for bottle_id in sorted(predictions):
        prediction = predictions[bottle_id]
        final = make_final_decision(prediction)
        rows.append({
            "bottle_id": bottle_id,
            "status": final["final_status"],
            "visual_status": prediction.get("status", "INCOMPLETE"),
            "visual_reasons": "; ".join(prediction.get("reasons", [])),
            "tare_kg": final["tare_kg"],
            "tare_status": final["tare_status"],
            "approval_until_year": final["approval_until_year"],
            "approval_status": final["approval_status"],
            "approval_valid": final["approval_valid"],
            "ready_frame": prediction.get("ready_frame"),
            "error": prediction.get("error", ""),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["bottle_id", "status", "visual_status", "visual_reasons",
                                      "tare_kg", "tare_status", "approval_until_year",
                                      "approval_status", "approval_valid", "ready_frame", "error"])
    csv_path = Path(output) / "bottles.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full three-camera modular GASS pipeline.")
    parser.add_argument("--weights", type=Path,
                        default=PROJECT_ROOT / "src" / "models" / "cosminCloud101Epochs.pt")
    parser.add_argument("--videos", type=Path, nargs=3, default=DEFAULT_VIDEOS,
                        help="Three videos ordered back_left back_right front")
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "modular_pipeline")
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--binary-classifier", type=Path)
    parser.add_argument("--binary-aggregation", choices=("max", "mean"))
    parser.add_argument("--nok-threshold", type=float)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--ocr-fields", type=Path)
    parser.add_argument("--ocr-selection-config", type=Path)
    parser.add_argument("--demo-dirt", action="store_true")
    parser.add_argument("--tracks-dir", type=Path)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--retrack", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args(argv)

    if (args.binary_aggregation is not None or args.nok_threshold is not None) and not args.binary_classifier:
        parser.error("--binary-aggregation and --nok-threshold require --binary-classifier.")
    if args.nok_threshold is not None and not 0 <= args.nok_threshold <= 1:
        parser.error("--nok-threshold must be finite and between 0 and 1.")
    if args.demo_dirt and not args.classifier:
        raise SystemExit("--demo-dirt requires --classifier.")
    if args.binary_classifier and (not args.classifier or args.demo_dirt):
        raise SystemExit("--binary-classifier requires --classifier and cannot use --demo-dirt.")
    if (args.ocr_fields or args.ocr_selection_config) and not args.ocr:
        raise SystemExit("--ocr-fields and --ocr-selection-config require --ocr.")

    device = select_device() if args.device == "auto" else args.device
    try:
        selection = load_selection(args.ocr_selection_config) if args.ocr else None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    optional = [path for path in (args.classifier, args.binary_classifier, args.ocr_fields) if path]
    for path in [args.weights, *args.videos, *optional]:
        if not path.is_file():
            raise SystemExit(f"File missing: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "output.mp4").exists() or (args.output / "predictions.json").exists():
        raise SystemExit("Output already exists; choose a new --output folder.")

    parser_stage = VideoParser()
    prepared = parser_stage.prepare(args.weights, args.videos, args.output,
                                    args.tracks_dir, args.retrack, device)
    predictor = None
    ocr = None

    if args.classifier:
        predictor = BottleClassifier(
            args.classifier, device, demo_dirt=args.demo_dirt,
            binary_checkpoint=args.binary_classifier,
            binary_aggregation=args.binary_aggregation or "max",
            nok_threshold=args.nok_threshold,
        )
    if args.ocr:
        try:
            ocr = BottleOCR(device, args.ocr_fields, selection)
        except ModuleNotFoundError as error:
            if error.name not in ("easyocr", "doctr"):
                raise
            raise SystemExit("Install requirements-ocr.txt in this Python environment.") from error

    start = prepared["start"]
    start = max(start, round(args.start_seconds * prepared["fps"]))
    end = prepared["end"]
    if args.seconds is not None:
        end = min(end, start + round(args.seconds * prepared["fps"]))

    args._predictor = predictor
    predictions = {}
    if args.classifier or args.ocr:
        print("Inspecting matched bottle triplets...", flush=True)
        predictions = inspect_crossings(predictor, prepared["captures"], prepared["tracks"],
                                        prepared["events"], prepared["offsets"], start, end, ocr)
        print(f"Prepared {len(predictions)} bottle results.", flush=True)

    for bottle_id, prediction in predictions.items():
        prediction.update(make_final_decision(prediction))

    output_video, _, final_end = parser_stage.render(args.videos, prepared, predictions,
                                                     args.output, args)

    summary = {
        "checkpoint": str(args.classifier) if args.classifier else None,
        "binary_checkpoint": str(args.binary_classifier) if args.binary_classifier else None,
        "decision_source": f"binary_{predictor.binary_aggregation}" if args.binary_classifier else "reasons" if args.classifier else None,
        "binary_aggregation": predictor.binary_aggregation if args.binary_classifier else None,
        "binary_threshold": predictor.binary_threshold if args.binary_classifier else None,
        "binary_project_policy": predictor.binary_project_policy if args.binary_classifier else None,
        "binary_policy_unverified": predictor.binary_policy_unverified if args.binary_classifier else None,
        "reason_schema_version": predictor.reason_schema_version if args.classifier else None,
        "legacy_reason_schema": predictor.reason_schema_version == 1 if args.classifier else None,
        "reason_order": list(predictor.reason_order) if args.classifier else None,
        "project_policy": predictor.project_policy if args.classifier else None,
        "videos": [str(path) for path in args.videos], "camera_order": list(VIDEO_CAMERAS),
        "offsets": prepared["offsets"], "start_frame": start, "end_frame": final_end,
        "fps": prepared["fps"], "status_scope": "visual", "ocr_enabled": args.ocr,
        "ocr_fields": str(args.ocr_fields) if args.ocr_fields else None,
        "ocr_selection": selection,
        "counts": inspect_counts(predictions, final_end - 1, start) if args.classifier else None,
        "predictions": list(predictions.values()),
        "mode": "modular three-module pipeline",
    }
    summary["demo_dirt_requested"] = args.demo_dirt
    (args.output / "predictions.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = export_bottle_csv(args.output, predictions)
    if args.classifier:
        print(f"Clip inspection counts: {summary['counts']}")
    else:
        print(f"Bottles crossed: {len(prepared['events'][0])}")
    print(f"Output video: {output_video}")
    print(f"Bottle CSV: {csv_path}")
    return output_video


if __name__ == "__main__":
    main()
