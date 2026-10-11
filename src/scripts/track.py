import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS

if __package__:
    from .config import DEVICE_CHOICES, IMAGE_SIZE, RUNS_DIR, select_device
else:
    from config import DEVICE_CHOICES, IMAGE_SIZE, RUNS_DIR, select_device


def main(argv=None):
    parser = argparse.ArgumentParser(description="Track cylinders and save an annotated video.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained detector weights.")
    parser.add_argument("--source", type=Path, required=True, help="Local video, e.g. data/primagaz_vids/conveyor.webm.")
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "tracking")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args(argv)

    weights, source = args.weights.resolve(), args.source.resolve()
    if not weights.is_file():
        raise SystemExit(f"Trained weights missing: {weights}")
    if not source.is_file():
        raise SystemExit(f"Video file missing: {source}")
    if source.suffix.lower().lstrip(".") not in VID_FORMATS:
        raise SystemExit("Source must be a video file; unrelated still images cannot establish tracks.")

    model = YOLO(str(weights))
    if model.task != "detect":
        raise SystemExit("Tracking requires bounding-box detection weights.")
    device = select_device() if args.device == "auto" else args.device
    print(f"Tracker: ByteTrack | Device: {device}")

    # Ultralytics reads consecutive frames, maintains IDs, and saves the video.
    results = model.track(
        source=str(source),
        tracker="bytetrack.yaml",
        conf=0.1,
        imgsz=IMAGE_SIZE,
        device=device,
        vid_stride=1,
        stream=True,
        save=True,
        project=str(args.output.resolve()),
        name="preview",
        exist_ok=False,
        verbose=False,
    )
    frame_count = 0
    # Consuming the stream runs inference without storing every frame in memory.
    for frame_count, _ in enumerate(results, start=1):
        pass

    run_dir = Path(model.predictor.save_dir)
    print(f"Processed frames: {frame_count}")
    print(f"Annotated video directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
