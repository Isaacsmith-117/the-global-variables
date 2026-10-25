# Synchronized tracking over three videos of the same conveyor (offline).
#
# Stage 1: track the cylinders in every video and save the boxes to a CSV.
# Stage 2: keep the real bottles (long enough tracks that move across the image).
# Stage 3: find the frame in which each real bottle crosses a vertical line.
# Stage 4: number the bottles in crossing order: the n-th bottle is bottle n in every camera.
# Stage 5: compute the delay between the cameras so the videos can be played in sync.
# Stage 6: draw the line, the boxes and the global IDs, and save output.mp4.
# With --classifier: classify matched triplets once, then overlay OK/NOK and reasons.
# With --ocr: read markings from those same crops and retain candidate fields.
#
# python src/scripts/tricam_offline.py
# python src/scripts/tricam_offline.py --videos a.mp4 b.mp4 c.mp4 --weights w.pt
# python src/scripts/tricam_offline.py --retrack      (ignore the saved CSVs, track again)
# python src/scripts/tricam_offline.py --no-display   (only save output.mp4)

import argparse
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image

if __package__:
    from .config import DEVICE_CHOICES, IMAGE_SIZE, PROJECT_ROOT, RUNS_DIR, select_device
else:
    from config import DEVICE_CHOICES, IMAGE_SIZE, PROJECT_ROOT, RUNS_DIR, select_device

VIDEO_CAMERAS = ("back_left", "back_right", "front")
DEFAULT_VIDEOS = [
    PROJECT_ROOT / "data" / "primagaz_vids" / "14_55" / f"14_55_{name}_cropped.mp4"
    for name in VIDEO_CAMERAS
]
MIN_FRAMES = 40      # tracks shorter than this are junk
MIN_MOVE_FRAC = 0.4  # a real bottle moves at least 40% of the image width
LINE_FRAC = 0.5      # the counting line sits at 50% of the image width
TILE_WIDTH = 640     # every video is shrunk to this width in output.mp4
GREY = (160, 160, 160)
COLORS = [
    (0, 220, 80), (255, 160, 0), (255, 60, 200), (0, 200, 255),
    (80, 80, 255), (200, 255, 0), (255, 255, 255), (160, 100, 255),
]


# ---------------------------------------------------------------- Stage 1
def track_to_csv(weights, video, csv_path, device=None):
    """Run YOLO + ByteTrack on one video and save every box to a CSV."""
    results = YOLO(str(weights)).track(
        source=str(video), tracker="bytetrack.yaml", conf=0.1, imgsz=IMAGE_SIZE,
        device=device or select_device(), stream=True, verbose=False,
    )
    rows = []
    for frame, r in enumerate(results):
        if frame % 500 == 0:
            print(f"  frame {frame}", flush=True)
        if r.boxes.id is None:  # nothing tracked in this frame
            continue
        for tid, (x1, y1, x2, y2), c in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist(),
                                            r.boxes.conf.tolist()):
            rows.append((frame, tid, round(x1), round(y1), round(x2), round(y2), round(c, 3)))
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
    df.to_csv(csv_path, index=False)
    return df


# ---------------------------------------------------------------- Stages 2 and 3
def crossings(df, width):
    """Return the real bottles of one camera as (track_id, cross_frame), in crossing order."""
    df = df.assign(cx=(df.x1 + df.x2) / 2)

    # Stage 2: keep tracks that last long enough and move far enough.
    stats = df.groupby("track_id").agg(frames=("frame", "count"),
                                       move=("cx", lambda s: s.max() - s.min()))
    good = stats[(stats.frames >= MIN_FRAMES) & (stats.move > MIN_MOVE_FRAC * width)].index
    d = df[df.track_id.isin(good)].sort_values(["track_id", "frame"])

    # Stage 3: the first frame in which the box centre is past the line.
    # sign flips the x axis when the bottles move right to left, so "progress" always grows.
    travel = d.groupby("track_id").cx.last() - d.groupby("track_id").cx.first()
    sign = -1 if travel.median() < 0 else 1
    line = sign * width * LINE_FRAC

    rows = []
    for tid, g in d.groupby("track_id"):
        progress = sign * g.cx
        if progress.iloc[0] >= line:
            continue  # already past the line in its first frame (the "zeroth" bottle)
        crossed = g.frame[progress >= line]
        if not crossed.empty:  # otherwise it was cut off at the end of the video
            rows.append((tid, int(crossed.iloc[0])))
    return sorted(rows, key=lambda row: row[1])


def inspect_crossings(classifier, caps, tracks, events, offsets, start, end, ocr=None):
    """Run optional visual/OCR checks on the same matched, full-resolution crops."""
    if len({len(e) for e in events}) != 1:
        raise ValueError("Camera counts differ; check bottle matching before inspection.")
    by_track = [dict(tuple(df.groupby("track_id"))) for df in tracks]
    predictions = {}
    for n in range(len(events[0])):
        views = [groups[events[i][n][0]] for i, groups in enumerate(by_track)]
        # Include bottles visible in any tile during this clip, including its edges.
        if max(int(g.frame.max()) - o for g, o in zip(views, offsets)) < start:
            continue
        if min(int(g.frame.min()) - o for g, o in zip(views, offsets)) >= end:
            continue
        gid = n + 1
        ready = max(events[i][n][1] - offsets[i] for i in range(3))
        result = {"bottle_id": gid, "status": "INCOMPLETE", "reasons": [], "ready_frame": ready}
        # A missed crossing can shift ordinal IDs: do not classify timing outliers.
        aligned = [events[i][n][1] - offsets[i] for i in range(3)]
        if max(aligned) - min(aligned) > 15:
            predictions[gid] = {**result, "error": "Camera crossing times do not match."}
            continue
        crops = {}
        boxes = {}
        for i, (camera, cap, rows) in enumerate(zip(VIDEO_CAMERAS, caps, views)):
            frame_index = events[i][n][1]
            box = rows[rows.frame == frame_index].iloc[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
            x2, y2 = min(width, int(box.x2)), min(height, int(box.y2))
            if x2 <= x1 or y2 <= y1:
                break
            crops[camera] = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            boxes[camera] = {"frame": frame_index, "track_id": int(box.track_id),
                             "box": [x1, y1, x2, y2]}
        if len(crops) == 3:
            if classifier is not None:
                result.update(classifier.predict(crops, gid))
            else:
                result["status"] = "NOT_CHECKED"
            result["crops"] = boxes
            if ocr is not None:
                result["ocr"] = {camera: ocr.read(image) for camera, image in crops.items()}
        else:
            result["error"] = "Could not read three usable crops."
        predictions[gid] = result
    return predictions


def inspection_counts(predictions, frame, start):
    """Count each final bottle decision once, when all three views are available."""
    completed = [p for p in predictions.values() if start <= p["ready_frame"] <= frame]
    return {status: sum(p["status"] == status for p in completed)
            for status in ("OK", "NOK", "INCOMPLETE")}


def draw_tag(frame, lines, x, y, color):
    """Readable labels at display resolution, kept inside the video tile."""
    lines = [part for line in lines for part in textwrap.wrap(line, width=48)]
    widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line in lines]
    width, height = max(widths) + 10, 22 * len(lines) + 6
    x = max(0, min(x, frame.shape[1] - width))
    y = max(0, min(y - height, frame.shape[0] - height))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (12, 12, 12), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 5, y + 19 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)


def ocr_label(reading):
    """Display tentative fields without treating extraction as an inspection decision."""
    if reading is None:
        return "OCR: unavailable"
    weights = ",".join(f"{v:g}" for v in reading["weight_kg_candidates"]) or "?"
    years = ",".join(map(str, reading["year_candidates"])) or "?"
    return f"OCR candidates | kg: {weights} | year: {years}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synchronized tracking over three videos.")
    parser.add_argument("--weights", type=Path,
                        default=PROJECT_ROOT / "src" / "models" / "cosminCloud101Epochs.pt")
    parser.add_argument("--videos", type=Path, nargs=3, default=DEFAULT_VIDEOS,
                        help="the 3 videos, ordered back_left back_right front")
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "tricam_offline")
    parser.add_argument("--classifier", type=Path, help="Optional trained three-view classifier checkpoint.")
    parser.add_argument("--ocr", action="store_true", help="Read candidate bottle markings locally.")
    parser.add_argument("--demo-dirt", action="store_true",
                        help="Include the untrained dirt output at a 0.5 cutoff for a demo.")
    parser.add_argument("--tracks-dir", type=Path, help="Read existing track CSVs from this folder.")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--seconds", type=float, help="Limit the output clip duration.")
    parser.add_argument("--retrack", action="store_true",
                        help="track again even if saved CSVs exist (use after changing weights)")
    parser.add_argument("--no-display", action="store_true", help="only save output.mp4")
    args = parser.parse_args(argv)
    device = select_device() if args.device == "auto" else args.device
    if args.start_seconds < 0 or (args.seconds is not None and args.seconds <= 0):
        raise SystemExit("Start must be nonnegative and clip duration must be positive.")
    if args.demo_dirt and not args.classifier:
        raise SystemExit("--demo-dirt requires --classifier.")

    for path in [args.weights, *args.videos, *([args.classifier] if args.classifier else [])]:
        if not path.is_file():
            raise SystemExit(f"File missing: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("output.mp4", "predictions.json")):
        raise SystemExit("Output already exists; choose a new --output folder.")

    caps = [cv2.VideoCapture(str(video)) for video in args.videos]
    if not all(cap.isOpened() for cap in caps):
        raise SystemExit("Could not open one of the videos.")
    width = [cap.get(cv2.CAP_PROP_FRAME_WIDTH) for cap in caps]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps <= 0 or any(abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 0.01 for cap in caps):
        raise SystemExit("The three recordings must have matching, positive frame rates.")

    # ------------------------------------------------------------ Stage 1: tracking
    # Slow, so the boxes are saved and reused on the next run (unless --retrack).
    tracks = []
    for video in args.videos:
        csv_path = (args.tracks_dir or args.output) / f"{video.stem}.csv"
        if args.tracks_dir and (not csv_path.is_file() or args.retrack):
            raise SystemExit("--tracks-dir requires existing CSVs and cannot be used with --retrack.")
        if csv_path.is_file() and not args.retrack:
            print(f"Using saved tracks {csv_path} (use --retrack to track again)")
            tracks.append(pd.read_csv(csv_path))
        else:
            print(f"Tracking {video.name} ...")
            tracks.append(track_to_csv(args.weights, video, csv_path, device))

    # ------------------------------------------------------------ Stages 2-3: real bottles + line crossing
    events = [crossings(df, w) for df, w in zip(tracks, width)]
    if any(not e for e in events):
        raise SystemExit("No complete bottle crossings in one or more cameras.")

    # ------------------------------------------------------------ Stage 4: global IDs
    # The n-th bottle crossing the line is bottle n in every camera.
    global_ids = [{tid: n for n, (tid, _) in enumerate(e, start=1)} for e in events]
    crossed_at = [dict(e) for e in events]  # track_id -> frame in which it crossed
    if len({len(e) for e in events}) > 1:
        print("Warning: the cameras counted a different number of bottles, so IDs can shift.")

    # ------------------------------------------------------------ Stage 5: delay between cameras
    # Same bottle, different crossing frame per camera. The median difference to the
    # first camera is the delay. Bottles far from that median hint at a matching error.
    common = min(len(e) for e in events)
    offsets = []
    for video, e in zip(args.videos, events):
        delays = [e[n][1] - events[0][n][1] for n in range(common)]
        offset = int(np.round(np.median(delays)))
        unusual = sum(abs(d - offset) > 15 for d in delays)
        offsets.append(offset)
        print(f"{video.stem}: {len(e)} bottles, delay {offset} frames, {unusual} unusual delays")

    # ------------------------------------------------------------ Stage 6: drawing
    # Boxes of the real bottles only, grouped by frame for fast lookup.
    boxes = [{f: g for f, g in df[df.track_id.isin(list(ids))].groupby("frame")}
             for df, ids in zip(tracks, global_ids)]

    # Play all videos from the moment every camera has a frame (camera frame = t + offset).
    totals = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    start = max(0, max(-o for o in offsets))
    start = max(start, round(args.start_seconds * fps))
    end = min(total - o for total, o in zip(totals, offsets))
    if args.seconds is not None:
        end = min(end, start + round(args.seconds * fps))
    if start >= end:
        raise SystemExit("Requested clip has no frames in the synchronized recordings.")
    predictions = {}
    predictor = None
    ocr = None
    if args.classifier:
        if __package__:
            from .classifier import BottleClassifier
        else:
            from classifier import BottleClassifier
        predictor = BottleClassifier(args.classifier, device, demo_dirt=args.demo_dirt)
    if args.ocr:
        if __package__:
            from .ocr import BottleOCR
        else:
            from ocr import BottleOCR
        try:
            ocr = BottleOCR(device)
        except ModuleNotFoundError as error:
            if error.name != "easyocr":
                raise
            raise SystemExit("Install requirements-ocr.txt in this Python environment.") from error
    if args.classifier or args.ocr:
        print("Inspecting matched bottle triplets...", flush=True)
        predictions = inspect_crossings(predictor, caps, tracks, events, offsets, start, end, ocr)
        print(f"Prepared {len(predictions)} bottle results.", flush=True)
    for cap, offset in zip(caps, offsets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + offset)

    # output.mp4 is a 2x2 grid: the 3 videos and an info panel.
    tile_h = 2 * round(TILE_WIDTH * caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT) / width[0] / 2)
    output = args.output / "output.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (2 * TILE_WIDTH, 2 * tile_h))
    if not writer.isOpened():
        raise SystemExit("Could not create the output video.")
    seen = set()  # global IDs that have crossed the line so far

    for t in range(start, end):
        tiles = []
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            if not ok:
                writer.release()
                for source in caps:
                    source.release()
                raise RuntimeError(f"Could not decode {args.videos[i]} at frame {t + offsets[i]}.")
            sx, sy = TILE_WIDTH / frame.shape[1], tile_h / frame.shape[0]
            frame = cv2.resize(frame, (TILE_WIDTH, tile_h))
            thick = 2

            # the counting line
            line_x = int(LINE_FRAC * frame.shape[1])
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), thick)

            # the boxes: grey "ID ?" before the line, coloured global ID after it
            f_now = t + offsets[i]
            frame_boxes = boxes[i].get(f_now)
            for r in [] if frame_boxes is None else frame_boxes.itertuples():
                gid = global_ids[i][r.track_id]
                passed = f_now >= crossed_at[i][r.track_id]
                if passed:
                    seen.add(gid)
                color = COLORS[gid % len(COLORS)] if passed else GREY
                label = f"ID {gid}" if passed else "ID ?"
                lines = [f"{label} {r.conf:.2f}"]
                prediction = predictions.get(gid)
                ready = prediction is not None and t >= prediction["ready_frame"]
                if args.classifier:
                    status = prediction["status"] if ready else "Inspecting"
                    color = {"OK": (60, 220, 70), "NOK": (70, 70, 255),
                             "INCOMPLETE": (0, 200, 255)}.get(status, GREY)
                    lines = [f"ID {gid} | Visual: {status}"]
                    if status == "NOK":
                        lines.append(", ".join(reason.replace("_", " ") for reason in prediction["reasons"]))
                if args.ocr:
                    reading = prediction.get("ocr", {}).get(VIDEO_CAMERAS[i]) if ready else None
                    lines.append(ocr_label(reading) if ready else "OCR: waiting for views")
                x1, y1, x2, y2 = int(r.x1 * sx), int(r.y1 * sy), int(r.x2 * sx), int(r.y2 * sy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                draw_tag(frame, lines, x1, y1, color)

            cv2.putText(frame, args.videos[i].stem, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(frame)

        # info panel
        panel = np.zeros((tile_h, TILE_WIDTH, 3), np.uint8)
        lines = ["THREE-CAMERA BOTTLE INSPECTION", f"Source time: {t / fps:.1f}s"]
        if args.classifier:
            counts = inspection_counts(predictions, t, start)
            lines += [f"Visual checked: {counts['OK'] + counts['NOK']}   OK: {counts['OK']}   NOK: {counts['NOK']}",
                      f"Incomplete: {counts['INCOMPLETE']}"]
            completed = [p for p in predictions.values() if start <= p["ready_frame"] <= t]
            if completed:
                latest = max(completed, key=lambda p: p["ready_frame"])
                lines.append(f"Latest bottle: {latest['bottle_id']} | {latest['status']}")
                lines += [reason.replace("_", " ") for reason in latest["reasons"]]
            unchecked = [r.replace("_", " ") for r in
                         ("major_damage", "heavy_dirt", "wrong_brand_or_model") if r not in predictor.enabled]
            lines.append("Unchecked: " + (", ".join(unchecked) if unchecked else "none"))
        else:
            lines.append(f"Bottles crossed: {len(seen)}")
        if args.ocr:
            lines.append("OCR: candidate markings; decision pending")
        for n, text in enumerate(lines):
            cv2.putText(panel, text, (20, 30 + 33 * n), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (235, 235, 235), 1, cv2.LINE_AA)

        grid = np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[2], panel])])
        writer.write(grid)
        if (t - start + 1) % 230 == 0:
            print(f"Rendered {t - start + 1}/{end - start} frames", flush=True)
        if not args.no_display:
            cv2.imshow("Tricam", grid)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to stop
                break

    writer.release()
    for cap in caps:
        cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if args.classifier or args.ocr:
        summary = {"checkpoint": str(args.classifier) if args.classifier else None,
                   "videos": [str(p) for p in args.videos],
                   "camera_order": VIDEO_CAMERAS, "offsets": offsets,
                   "start_frame": start, "end_frame": t + 1, "fps": fps,
                   "status_scope": "visual", "ocr_enabled": args.ocr,
                   "counts": inspection_counts(predictions, t, start) if args.classifier else None,
                   "predictions": list(predictions.values()), "mode": "preprocessed integration demo"}
        summary["demo_dirt_requested"] = args.demo_dirt
        (args.output / "predictions.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.classifier:
        print(f"Clip inspection counts: {summary['counts']}")
    else:
        print(f"Bottles crossed: {len(seen)}")
    print(f"Output video: {output}")
    return output


if __name__ == "__main__":
    main()
