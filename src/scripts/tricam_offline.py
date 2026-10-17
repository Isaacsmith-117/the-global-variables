# Synchronized tracking over three videos of the same conveyor (Emilija's method,
# branch 08-Emilija-tracking-over-multiple-videos):
# 1. Track the cylinders in every video with ByteTrack and save the boxes to a CSV.
# 2. Keep the real bottles: tracks that last long enough and move across the image.
# 3. Number the bottles in the order they cross a line in the middle of the image:
#    the n-th bottle that crosses the line is the same bottle in every camera.
# 4. Align the videos by the delay between the cameras and save output.mp4:
#    the 3 annotated videos and a panel with the bottles seen so far and in view now.
# It needs the whole videos first, so it is not live; src/scripts/tricam.py is the live version.
#
# python src/scripts/tricam_offline.py
# python src/scripts/tricam_offline.py --videos a.mp4 b.mp4 c.mp4
# python src/scripts/tricam_offline.py --no-display   (only save output.mp4)

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

if __package__:
    from .config import IMAGE_SIZE, PROJECT_ROOT, RUNS_DIR, select_device
else:
    from config import IMAGE_SIZE, PROJECT_ROOT, RUNS_DIR, select_device

DEFAULT_VIDEOS = [
    PROJECT_ROOT / "data" / "video" / f"14_55_{name}_cropped.mp4"
    for name in ("back_left", "back_right", "front")
]
MIN_FRAMES = 40      # shorter tracks are junk
MIN_MOVE_FRAC = 0.4  # a real bottle moves at least 40% of the image width
LINE_FRAC = 0.5      # the line sits at 50% of the width
TILE_WIDTH = 640     # every video is shrunk to this width in output.mp4
COLORS = [
    (0, 220, 80), (255, 160, 0), (255, 60, 200), (0, 200, 255),
    (80, 80, 255), (200, 255, 0), (255, 255, 255), (160, 100, 255),
]


def track_to_csv(weights, video, csv_path):
    """Track the cylinders in one video and save every box to a CSV."""
    results = YOLO(str(weights)).track(
        source=str(video), tracker="bytetrack.yaml", conf=0.1, imgsz=IMAGE_SIZE,
        device=select_device(), stream=True, verbose=False,
    )
    rows = []
    for frame, r in enumerate(results):
        if r.boxes.id is None:  # no tracked objects in this frame
            continue
        for tid, (x1, y1, x2, y2), c in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist(),
                                            r.boxes.conf.tolist()):
            rows.append((frame, tid, round(x1), round(y1), round(x2), round(y2), round(c, 3)))
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
    df.to_csv(csv_path, index=False)
    return df


def crossings(df, width):
    """Return the real bottles of one camera as (track_id, cross_frame), in crossing order."""
    df = df.assign(cx=(df.x1 + df.x2) / 2)
    stats = df.groupby("track_id").agg(frames=("frame", "count"),
                                       move=("cx", lambda s: s.max() - s.min()))
    good = stats[(stats.frames >= MIN_FRAMES) & (stats.move > MIN_MOVE_FRAC * width)].index
    d = df[df.track_id.isin(good)].sort_values(["track_id", "frame"])

    # Which way do the bottles travel? Flip the sign so "progress" always increases.
    travel = d.groupby("track_id").cx.last() - d.groupby("track_id").cx.first()
    sign = -1 if travel.median() < 0 else 1
    line = sign * width * LINE_FRAC

    rows = []
    for tid, g in d.groupby("track_id"):
        progress = sign * g.cx
        if progress.iloc[0] >= line:
            continue  # already past the line in its first frame
        crossed = g.frame[progress >= line]
        if not crossed.empty:  # otherwise it was cut off at the end of the video
            rows.append((tid, int(crossed.iloc[0])))
    return sorted(rows, key=lambda row: row[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synchronized tracking over three videos.")
    parser.add_argument("--weights", type=Path,
                        default=PROJECT_ROOT / "src" / "models" / "cosminCloud101Epochs.pt")
    parser.add_argument("--videos", type=Path, nargs=3, default=DEFAULT_VIDEOS,
                        help="the 3 videos of the same conveyor")
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "tricam_offline")
    parser.add_argument("--no-display", action="store_true", help="only save output.mp4")
    args = parser.parse_args(argv)

    for path in [args.weights, *args.videos]:
        if not path.is_file():
            raise SystemExit(f"File missing: {path}")
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Tracking is slow, so the boxes are saved and reused on the next run.
    caps = [cv2.VideoCapture(str(video)) for video in args.videos]
    tracks = []
    for video in args.videos:
        csv_path = args.output / f"{video.stem}.csv"
        if csv_path.is_file():
            print(f"Using saved tracks {csv_path} (delete it to track again)")
            tracks.append(pd.read_csv(csv_path))
        else:
            print(f"Tracking {video.name} ...")
            tracks.append(track_to_csv(args.weights, video, csv_path))

    # 2-3. The n-th bottle that crosses the line gets ID n in every camera.
    events = [crossings(df, cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for df, cap in zip(tracks, caps)]
    global_ids = [{tid: n for n, (tid, _) in enumerate(e, start=1)} for e in events]

    # 4. Delay of every camera compared with the first one: the median difference
    #    between the frames in which the same bottle crosses the line.
    common = min(len(e) for e in events)
    offsets = []
    for video, e in zip(args.videos, events):
        delays = [e[n][1] - events[0][n][1] for n in range(common)]
        offset = int(np.median(delays)) if delays else 0
        unusual = sum(abs(d - offset) > 15 for d in delays)
        offsets.append(offset)
        print(f"{video.stem}: {len(e)} bottles, delay {offset} frames, "
              f"{unusual} bottles with an unusual delay")
    if len({len(e) for e in events}) > 1:
        print("Warning: the cameras counted a different number of bottles, so IDs can shift.")

    # Only the real bottles are drawn, with their global ID.
    boxes = [{f: g for f, g in df[df.track_id.isin(ids)].groupby("frame")}
             for df, ids in zip(tracks, global_ids)]
    totals = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    start = max(0, max(-o for o in offsets))  # camera frame = t + offset must be >= 0
    end = min(total - o for total, o in zip(totals, offsets))
    for cap, offset in zip(caps, offsets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + offset)

    # output.mp4 is split in 4 parts: the 3 annotated videos and an info panel.
    tile_h = round(TILE_WIDTH * caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)
                   / caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    output = args.output / "output.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             caps[0].get(cv2.CAP_PROP_FPS), (2 * TILE_WIDTH, 2 * tile_h))
    seen = set()
    for t in range(start, end):
        tiles, in_view = [], set()
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((tile_h, TILE_WIDTH, 3), np.uint8)
            s = frame.shape[1] / TILE_WIDTH  # The frame is shrunk later, so scale the labels up.
            frame_boxes = boxes[i].get(t + offsets[i])
            for r in [] if frame_boxes is None else frame_boxes.itertuples():
                gid = global_ids[i][r.track_id]
                in_view.add(gid)
                color = COLORS[gid % len(COLORS)]
                x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, round(3 * s)))
                cv2.putText(frame, f"ID {gid} {r.conf:.2f}",
                            (x1, max(int(40 * s), y1 - int(10 * s))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3 * s, color, max(2, round(3 * s)))
            cv2.putText(frame, args.videos[i].stem, (int(10 * s), frame.shape[0] - int(15 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255), max(1, round(2 * s)))
            tiles.append(cv2.resize(frame, (TILE_WIDTH, tile_h)))
        seen |= in_view

        panel = np.zeros((tile_h, TILE_WIDTH, 3), np.uint8)
        lines = [f"Frame {t}", f"Bottles seen so far: {len(seen)}",
                 f"Bottles in view now: {len(in_view)}"]
        for n, line in enumerate(lines):
            cv2.putText(panel, line, (20, 50 + 45 * n), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[2], panel])])
        writer.write(grid)
        if not args.no_display:
            cv2.imshow("Tricam", grid)
            if cv2.waitKey(1) == ord("q"):  # Press q to stop.
                break

    writer.release()
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Bottles seen: {len(seen)}")
    print(f"Output video: {output}")


if __name__ == "__main__":
    main()
