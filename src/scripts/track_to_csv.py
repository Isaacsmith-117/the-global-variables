import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

if __package__:
    from .config import IMAGE_SIZE, select_device
else:
    from config import IMAGE_SIZE, select_device


def main(argv=None):
    parser = argparse.ArgumentParser(description="Track and save detections to CSV.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tracks"))
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / f"{args.source.stem}.csv"

    model = YOLO(str(args.weights))
    results = model.track(
        source=str(args.source),
        tracker="bytetrack.yaml",
        conf=0.1,
        imgsz=IMAGE_SIZE,
        device=select_device(),
        stream=True,
        save=False,      # no video, so it runs faster
        verbose=False,
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
        for frame_idx, r in enumerate(results):
            if r.boxes.id is None:   # no tracked objects in this frame
                continue
            ids = r.boxes.id.int().tolist()
            boxes = r.boxes.xyxy.tolist()
            confs = r.boxes.conf.tolist()
            for tid, (x1, y1, x2, y2), c in zip(ids, boxes, confs):
                writer.writerow([frame_idx, tid, round(x1), round(y1), round(x2), round(y2), round(c, 3)])
            if frame_idx % 200 == 0:
                print(f"Frame {frame_idx}", flush=True)

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()