# Synchronized tracking over three videos of the same conveyor (offline).
#
# Stage 1: track the cylinders in every video and save the boxes to a CSV.
# Stage 2: keep the real bottles (long enough tracks that move across the image).
# Stage 3: find the frame in which each real bottle crosses a vertical line.
# Stage 4: number the bottles in crossing order: the n-th bottle is bottle n in every camera.
# Stage 5: compute the delay between the cameras so the videos can be played in sync.
# Stage 6: draw the line, the boxes and the global IDs, and save output.mp4.
#
# python src/scripts/tricam_offline.py
# python src/scripts/tricam_offline.py --videos a.mp4 b.mp4 c.mp4 --weights w.pt
# python src/scripts/tricam_offline.py --retrack      (ignore the saved CSVs, track again)
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
def track_to_csv(weights, video, csv_path):
    """Run YOLO + ByteTrack on one video and save every box to a CSV."""
    results = YOLO(str(weights)).track(
        source=str(video), tracker="bytetrack.yaml", conf=0.1, imgsz=IMAGE_SIZE,
        device=select_device(), stream=True, verbose=False,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synchronized tracking over three videos.")
    parser.add_argument("--weights", type=Path,
                        default=PROJECT_ROOT / "src" / "models" / "cosminCloud101Epochs.pt")
    parser.add_argument("--videos", type=Path, nargs=3, default=DEFAULT_VIDEOS,
                        help="the 3 videos of the same conveyor")
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "tricam_offline")
    parser.add_argument("--retrack", action="store_true",
                        help="track again even if saved CSVs exist (use after changing weights)")
    parser.add_argument("--no-display", action="store_true", help="only save output.mp4")
    args = parser.parse_args(argv)

    for path in [args.weights, *args.videos]:
        if not path.is_file():
            raise SystemExit(f"File missing: {path}")
    args.output.mkdir(parents=True, exist_ok=True)

    caps = [cv2.VideoCapture(str(video)) for video in args.videos]
    if not all(cap.isOpened() for cap in caps):
        raise SystemExit("Could not open one of the videos.")
    width = [cap.get(cv2.CAP_PROP_FRAME_WIDTH) for cap in caps]

    # ------------------------------------------------------------ Stage 1: tracking
    # Slow, so the boxes are saved and reused on the next run (unless --retrack).
    tracks = []
    for video in args.videos:
        csv_path = args.output / f"{video.stem}.csv"
        if csv_path.is_file() and not args.retrack:
            print(f"Using saved tracks {csv_path} (use --retrack to track again)")
            tracks.append(pd.read_csv(csv_path))
        else:
            print(f"Tracking {video.name} ...")
            tracks.append(track_to_csv(args.weights, video, csv_path))

    # ------------------------------------------------------------ Stages 2-3: real bottles + line crossing
    events = [crossings(df, w) for df, w in zip(tracks, width)]

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
    end = min(total - o for total, o in zip(totals, offsets))
    for cap, offset in zip(caps, offsets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + offset)

    # output.mp4 is a 2x2 grid: the 3 videos and an info panel.
    tile_h = round(TILE_WIDTH * caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT) / width[0])
    output = args.output / "output.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             caps[0].get(cv2.CAP_PROP_FPS), (2 * TILE_WIDTH, 2 * tile_h))
    seen = set()  # global IDs that have crossed the line so far

    for t in range(start, end):
        tiles = []
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((tile_h, TILE_WIDTH, 3), np.uint8)
            s = frame.shape[1] / TILE_WIDTH  # labels are drawn on the big frame, so scale them up
            thick = max(2, round(3 * s))

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
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, thick)
                cv2.putText(frame, f"{label} {r.conf:.2f}",
                            (int(r.x1), max(int(40 * s), int(r.y1) - int(10 * s))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3 * s, color, thick)

            cv2.putText(frame, args.videos[i].stem, (int(10 * s), frame.shape[0] - int(15 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255), max(1, round(2 * s)))
            tiles.append(cv2.resize(frame, (TILE_WIDTH, tile_h)))

        # info panel
        panel = np.zeros((tile_h, TILE_WIDTH, 3), np.uint8)
        for n, text in enumerate([f"Frame {t}", f"Bottles crossed: {len(seen)}"]):
            cv2.putText(panel, text, (20, 50 + 45 * n), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

        grid = np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[2], panel])])
        writer.write(grid)
        if not args.no_display:
            cv2.imshow("Tricam", grid)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to stop
                break

    writer.release()
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Bottles crossed: {len(seen)}")
    print(f"Output video: {output}")


if __name__ == "__main__":
    main()