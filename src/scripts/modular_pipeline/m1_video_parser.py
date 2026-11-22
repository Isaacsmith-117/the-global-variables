"""Bottle detection, three-camera matching, crop extraction and video rendering."""

import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import torch

try:
    from .pipeline_config import (
        COLORS, GREY, IMAGE_SIZE, LINE_FRAC, MAX_CROSSING_SPREAD_FRAMES,
        MIN_MOVE_FRAC, MIN_TRACK_FRAMES, REASON_LABELS, TILE_WIDTH,
        TRACK_CONFIDENCE, TRACKER, VIDEO_CAMERAS,
    )
except ImportError:
    from pipeline_config import (
        COLORS, GREY, IMAGE_SIZE, LINE_FRAC, MAX_CROSSING_SPREAD_FRAMES,
        MIN_MOVE_FRAC, MIN_TRACK_FRAMES, REASON_LABELS, TILE_WIDTH,
        TRACK_CONFIDENCE, TRACKER, VIDEO_CAMERAS,
    )


MIN_FRAMES = MIN_TRACK_FRAMES


def select_device():
    """Use an available accelerator or fall back to CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def track_to_csv(weights, video, csv_path, device=None, max_seconds=None):
    """Run YOLO + ByteTrack and save detected track boxes for the requested prefix."""
    weights, video, csv_path = Path(weights), Path(video), Path(csv_path)
    max_frames = None
    if max_seconds is not None:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if fps > 0:
            max_frames = max(0, round(float(max_seconds) * fps))

    results = YOLO(str(weights)).track(
        source=str(video), tracker=TRACKER, conf=TRACK_CONFIDENCE, imgsz=IMAGE_SIZE,
        device=device or select_device(), stream=True, verbose=False,
    )
    rows = []
    for frame, result in enumerate(results):
        if max_frames is not None and frame >= max_frames:
            break
        if frame % 500 == 0:
            print(f"  frame {frame}", flush=True)
        if result.boxes.id is None:
            continue
        for track_id, (x1, y1, x2, y2), confidence in zip(
            result.boxes.id.int().tolist(), result.boxes.xyxy.tolist(), result.boxes.conf.tolist()
        ):
            rows.append((frame, track_id, round(x1), round(y1), round(x2), round(y2), round(confidence, 3)))
    frame_tracks = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame_tracks.to_csv(csv_path, index=False)
    return frame_tracks


def crossings(frame_tracks, width):
    """Return (track_id, cross_frame) pairs for real, moving tracks."""
    frame_tracks = frame_tracks.assign(cx=(frame_tracks.x1 + frame_tracks.x2) / 2)
    stats = frame_tracks.groupby("track_id").agg(
        frames=("frame", "count"), move=("cx", lambda values: values.max() - values.min())
    )
    good = stats[(stats.frames >= MIN_FRAMES) & (stats.move > MIN_MOVE_FRAC * width)].index
    tracks = frame_tracks[frame_tracks.track_id.isin(good)].sort_values(["track_id", "frame"])
    travel = tracks.groupby("track_id").cx.last() - tracks.groupby("track_id").cx.first()
    sign = -1 if travel.median() < 0 else 1
    line = sign * width * LINE_FRAC

    result = []
    for track_id, group in tracks.groupby("track_id"):
        progress = sign * group.cx
        if progress.iloc[0] >= line:
            continue
        crossed = group.frame[progress >= line]
        if not crossed.empty:
            result.append((track_id, int(crossed.iloc[0])))
    return sorted(result, key=lambda item: item[1])


def draw_tag(frame, lines, x, y, color):
    lines = [part for line in lines for part in textwrap.wrap(line, width=48)]
    widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line in lines]
    width, height = max(widths) + 10, 22 * len(lines) + 6
    x = max(0, min(x, frame.shape[1] - width))
    y = max(0, min(y - height, frame.shape[0] - height))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (12, 12, 12), -1)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x + 5, y + 19 + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def ocr_label(reading):
    if reading is None:
        return "OCR: unavailable"
    fields = [f"{name}: {'?' if field['value'] is None else field['value']}"
              for name, field in reading["fields"].items()]
    return "OCR | " + (" | ".join(fields) or "no fields")


def bottle_table_row(prediction):
    """Build compact inspection values for the on-video bottle table."""
    final_status = prediction.get("final_status", prediction.get("status", "INCOMPLETE"))
    visual_status = prediction.get("status", "INCOMPLETE")

    tare_field = prediction.get("ocr_fields", {}).get("tare_kg", {})
    tare_value = tare_field.get("value") if tare_field.get("status") == "accepted" else None
    tare_status = tare_field.get("status", "not_checked")
    tare = f"{tare_value:g}" if isinstance(tare_value, (int, float)) else "-"

    approval_field = prediction.get("ocr_fields", {}).get("approval_until_date", {})
    approval_status = approval_field.get("status", "not_checked")
    approval_value = approval_field.get("value")
    try:
        year = str(int(str(approval_value)[:4])) if approval_status == "accepted" else "-"
    except (TypeError, ValueError):
        year = "-"

    if tare_status == "accepted" and approval_status == "accepted":
        ocr_status = "accepted"
    elif tare_status == "missing" or approval_status == "missing":
        ocr_status = "missing"
    elif tare_status == "not_checked" and approval_status == "not_checked":
        ocr_status = "not_checked"
    else:
        ocr_status = "review"

    reasons = ";".join(prediction.get("reasons", [])[:2])
    visual = visual_status if not reasons else f"{visual_status}:{reasons}"
    short_status = {
        "NOT_CHECKED": "UNCHECKED",
        "INCOMPLETE": "INCOMPLETE",
    }.get(final_status, final_status)
    visual_status = {
        "NOT_CHECKED": "UNCHECKED",
    }.get(visual_status, visual_status)
    if reasons:
        visual = f"{visual_status}:{';'.join(reasons[:2])}"
    else:
        visual = visual_status

    return short_status, visual, tare, year, ocr_status


def draw_bottle_table(panel, predictions, synchronized_frame, start, table_top=None):
    """Draw newest-first completed bottle results, highlighting the most recent row."""
    completed = [prediction for prediction in predictions.values()
                 if start <= prediction.get("ready_frame", synchronized_frame + 1) <= synchronized_frame]
    completed.sort(key=lambda prediction: prediction.get("ready_frame", 0), reverse=True)

    if table_top is None:
        table_top = 190
    if not completed:
        cv2.putText(panel, "BOTTLES: waiting for crossings", (20, table_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
        return

    cv2.putText(panel, "BOTTLE INSPECTIONS (newest first)", (20, table_top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    header_y = table_top + 25
    columns = ((20, "ID"), (55, "FINAL"), (175, "VISUAL"),
               (350, "TARE"), (420, "YEAR"), (495, "OCR"))
    for x, label in columns:
        cv2.putText(panel, label, (x, header_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1, cv2.LINE_AA)

    row_height = 25
    first_row_y = header_y + 18
    max_rows = max(0, (panel.shape[0] - first_row_y) // row_height)
    latest_ready = completed[0].get("ready_frame") if completed else None

    for index, prediction in enumerate(completed[:max_rows]):
        y = first_row_y + index * row_height
        final_status, visual, tare, year, ocr_status = bottle_table_row(prediction)
        latest = prediction.get("ready_frame") == latest_ready

        if latest:
            cv2.rectangle(panel, (10, y - 17), (panel.shape[1] - 10, y + 5),
                          (45, 45, 20), -1)
            cv2.rectangle(panel, (10, y - 17), (16, y + 5),
                          (0, 220, 255), -1)

        status_color = {"OK": (60, 220, 70), "NOK": (70, 70, 255),
                        "INCOMPLETE": (0, 200, 255)}.get(final_status, (180, 180, 180))
        cv2.putText(panel, str(prediction.get("bottle_id", "-")), (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, str(final_status), (55, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1, cv2.LINE_AA)

        visual_text = textwrap.shorten(str(visual), width=20, placeholder="...")
        cv2.putText(panel, visual_text, (175, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{tare} kg" if tare != "-" else "-", (350, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, year, (420, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, ocr_status, (495, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


def reason_labels(prediction):
    lines = []
    if prediction["reasons"]:
        text = ", ".join(REASON_LABELS.get(reason, reason.replace("_", " "))
                         for reason in prediction["reasons"])
        lines.append("Supporting: " + text if "reason_status" in prediction else text)
    if prediction.get("reason_schema_version") == 1:
        lines.append("Legacy three-reason schema")
    if prediction.get("binary_policy_unverified"):
        lines.append("Binary policy unverified")
    if prediction.get("reason_unresolved"):
        lines.append("Reason unresolved")
    if prediction.get("reason_disagreement"):
        lines.append("Binary/reason disagreement")
    return lines


class VideoParser:
    """Detect, synchronize and render three conveyor recordings."""

    def prepare(self, weights, videos, output, tracks_dir=None, retrack=False,
                device="cpu", max_seconds=None):
        weights, videos = Path(weights), [Path(video) for video in videos]
        output = Path(output)
        for weight, video in [(weights, None), *[(None, video) for video in videos]]:
            if weight is not None and not weight.is_file():
                raise SystemExit("Trained weights missing: " + str(weight))
            if video is not None and not video.is_file():
                raise SystemExit("Video file missing: " + str(video))

        output.mkdir(parents=True, exist_ok=True)
        captures = [cv2.VideoCapture(str(video)) for video in videos]
        if not all(capture.isOpened() for capture in captures):
            raise SystemExit("Could not open one of the videos.")
        widths = [capture.get(cv2.CAP_PROP_FRAME_WIDTH) for capture in captures]
        fps = captures[0].get(cv2.CAP_PROP_FPS)
        if fps <= 0 or any(abs(capture.get(cv2.CAP_PROP_FPS) - fps) > 0.01 for capture in captures):
            raise SystemExit("The three recordings must have matching, positive frame rates.")

        tracks = []
        duration_suffix = ""
        if max_seconds is not None:
            duration_suffix = f"_first_{max_seconds:g}s"
        for video in videos:
            csv_path = (tracks_dir or output) / f"{video.stem}{duration_suffix}.csv"
            if tracks_dir and (not csv_path.is_file() or retrack):
                raise SystemExit("--tracks-dir requires existing prefix-track CSVs and cannot be used with --retrack.")
            if csv_path.is_file() and not retrack:
                print(f"Using saved tracks {csv_path} (use --retrack to track again)")
                tracks.append(pd.read_csv(csv_path))
            else:
                limit = f" for the first {max_seconds:g} seconds" if max_seconds is not None else ""
                print(f"Tracking {video.name}{limit} ...")
                tracks.append(track_to_csv(weights, video, csv_path, device, max_seconds))

        events = [crossings(frame_tracks, width) for frame_tracks, width in zip(tracks, widths)]
        if any(not camera_events for camera_events in events):
            raise SystemExit("No complete bottle crossings in one or more cameras.")

        global_ids = [{track_id: bottle_id for bottle_id, (track_id, _) in enumerate(camera_events, start=1)}
                      for camera_events in events]
        crossed_at = [dict(camera_events) for camera_events in events]
        if len({len(camera_events) for camera_events in events}) > 1:
            print("Warning: the cameras counted a different number of bottles, so IDs can shift.")

        offsets = []
        common = min(len(camera_events) for camera_events in events)
        for video, camera_events in zip(videos, events):
            delays = [camera_events[index][1] - events[0][index][1] for index in range(common)]
            offset = int(np.round(np.median(delays)))
            unusual = sum(abs(delay - offset) > MAX_CROSSING_SPREAD_FRAMES for delay in delays)
            offsets.append(offset)
            print(f"{video.stem}: {len(camera_events)} bottles, delay {offset} frames, {unusual} unusual delays")

        boxes = [{frame: group for frame, group in frame_tracks[frame_tracks.track_id.isin(ids)].groupby("frame")}
                 for frame_tracks, ids in zip(tracks, global_ids)]
        totals = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
        start = max(0, max(-offset for offset in offsets))
        end = min(total - offset for total, offset in zip(totals, offsets))
        return {
            "captures": captures, "fps": fps, "widths": widths, "tracks": tracks,
            "events": events, "global_ids": global_ids, "crossed_at": crossed_at,
            "offsets": offsets, "boxes": boxes, "start": start, "end": end,
        }

    def render(self, videos, prepared, predictions, output, args):
        """Create the existing 2x2 annotated output.mp4."""
        captures = prepared["captures"]
        fps = prepared["fps"]
        widths = prepared["widths"]
        offsets = prepared["offsets"]
        boxes = prepared["boxes"]
        global_ids = prepared["global_ids"]
        crossed_at = prepared["crossed_at"]
        start = max(prepared["start"], round(args.start_seconds * fps))
        end = prepared["end"]
        if args.seconds is not None:
            end = min(end, start + round(args.seconds * fps))
        if start >= end:
            raise SystemExit("Requested clip has no frames in the synchronized recordings.")

        for capture, offset in zip(captures, offsets):
            capture.set(cv2.CAP_PROP_POS_FRAMES, start + offset)
        tile_height = 2 * round(TILE_WIDTH * captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT) / widths[0] / 2)
        output_video = Path(output) / "output.mp4"
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (2 * TILE_WIDTH, 2 * tile_height))
        if not writer.isOpened():
            raise SystemExit("Could not create the output video.")

        seen = set()
        predictor = args._predictor
        for synchronized_frame in range(start, end):
            tiles = []
            for camera_index, capture in enumerate(captures):
                readable, frame = capture.read()
                if not readable:
                    writer.release()
                    for source in captures:
                        source.release()
                    raise RuntimeError(f"Could not decode {videos[camera_index]} at frame {synchronized_frame + offsets[camera_index]}.")
                sx = TILE_WIDTH / frame.shape[1]
                sy = tile_height / frame.shape[0]
                frame = cv2.resize(frame, (TILE_WIDTH, tile_height))
                cv2.line(frame, (int(LINE_FRAC * frame.shape[1]), 0),
                         (int(LINE_FRAC * frame.shape[1]), frame.shape[0]), (0, 255, 255), 2)
                source_frame = synchronized_frame + offsets[camera_index]
                frame_boxes = boxes[camera_index].get(source_frame)
                for row in [] if frame_boxes is None else frame_boxes.itertuples():
                    bottle_id = global_ids[camera_index][row.track_id]
                    passed = source_frame >= crossed_at[camera_index][row.track_id]
                    if passed:
                        seen.add(bottle_id)
                    color = COLORS[bottle_id % len(COLORS)] if passed else GREY
                    label = f"ID {bottle_id}" if passed else "ID ?"
                    lines = [f"{label} {row.conf:.2f}"]
                    prediction = predictions.get(bottle_id)
                    ready = prediction is not None and synchronized_frame >= prediction["ready_frame"]
                    if args.classifier:
                        status = prediction["status"] if ready else "Inspecting"
                        color = {"OK": (60, 220, 70), "NOK": (70, 70, 255),
                                 "INCOMPLETE": (0, 200, 255)}.get(status, GREY)
                        lines = [f"ID {bottle_id} | Visual: {status}"]
                        if ready:
                            lines += reason_labels(prediction)
                    if args.ocr:
                        reading = {"fields": prediction.get("ocr_fields", {})} if ready else None
                        lines.append(ocr_label(reading) if ready else "OCR: waiting for views")
                    x1, y1, x2, y2 = int(row.x1 * sx), int(row.y1 * sy), int(row.x2 * sx), int(row.y2 * sy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_tag(frame, lines, x1, y1, color)
                cv2.putText(frame, videos[camera_index].stem, (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(frame)

            panel = np.zeros((tile_height, TILE_WIDTH, 3), np.uint8)
            panel_lines = ["THREE-CAMERA BOTTLE INSPECTION", f"Source time: {synchronized_frame / fps:.1f}s"]
            if args.binary_classifier:
                panel_lines.append(f"Binary: {predictor.binary_aggregation}, cutoff {predictor.binary_threshold:g}")
            if args.classifier:
                completed = [prediction for prediction in predictions.values()
                             if start <= prediction["ready_frame"] <= synchronized_frame]
                panel_lines += [f"Visual checked: {sum(p['status'] in ('OK', 'NOK') for p in completed)}   "
                                f"OK: {sum(p['status'] == 'OK' for p in completed)}   "
                                f"NOK: {sum(p['status'] == 'NOK' for p in completed)}",
                                f"Incomplete: {sum(p['status'] == 'INCOMPLETE' for p in completed)}"]
                if predictor is not None:
                    unchecked = [REASON_LABELS.get(reason, reason.replace("_", " "))
                                 for reason in predictor.reason_order if reason not in predictor.enabled]
                    schema_label = "1 (legacy)" if predictor.reason_schema_version == 1 else str(predictor.reason_schema_version)
                    panel_lines.append(f"Reason schema: {schema_label}")
                    panel_lines.append("Unchecked: " + (", ".join(unchecked) if unchecked else "none"))
            else:
                panel_lines.append(f"Bottles crossed: {len(seen)}")
            panel_lines = [part for line in panel_lines for part in textwrap.wrap(line, width=48)]
            for index, text in enumerate(panel_lines):
                cv2.putText(panel, text, (20, 26 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (235, 235, 235), 1, cv2.LINE_AA)

            table_predictions = dict(predictions)
            for bottle_id in seen:
                if bottle_id in table_predictions:
                    continue
                ready_frames = []
                for camera_index, mapping in enumerate(global_ids):
                    for track_id, mapped_id in mapping.items():
                        if mapped_id == bottle_id:
                            ready_frames.append(crossed_at[camera_index][track_id] - offsets[camera_index])
                if ready_frames:
                    table_predictions[bottle_id] = {
                        "bottle_id": bottle_id,
                        "ready_frame": max(ready_frames),
                        "status": "NOT_CHECKED",
                        "final_status": "NOT_CHECKED",
                        "ocr_fields": {},
                        "reasons": [],
                    }

            table_top = 26 + 21 * len(panel_lines) + 38
            draw_bottle_table(panel, table_predictions, synchronized_frame, start, table_top)

            grid = np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[2], panel])])
            writer.write(grid)
            if (synchronized_frame - start + 1) % 230 == 0:
                print(f"Rendered {synchronized_frame - start + 1}/{end - start} frames", flush=True)
            if not args.no_display:
                try:
                    cv2.imshow("Tricam", grid)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("Display unavailable (OpenCV GUI support is missing); continuing without a window.", flush=True)
                    args.no_display = True

        writer.release()
        for capture in captures:
            capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        return output_video, start, end
