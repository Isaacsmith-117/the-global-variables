import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO = PROJECT_ROOT / "data" / "video_banda.mp4"


def main():
    parser = argparse.ArgumentParser(description="Track gas cylinders in a video.")
    parser.add_argument("--weights", required=True, help="Path to best.pt.")
    args = parser.parse_args()

    # Detect the cylinders in each frame and give every cylinder an ID
    # that stays the same while it moves on the conveyor belt.
    model = YOLO(args.weights)
    model.track(
        source=str(VIDEO),
        tracker="bytetrack.yaml",  # Or "botsort.yaml".
        persist=True,  # Keep the IDs from one frame to the next.
        conf=0.4,  # Ignore detections below 40% confidence.
        vid_stride=3,  # Process 1 frame out of 3.
        save=True,  # Save the video with boxes and IDs.
        project=str(PROJECT_ROOT / "runs"),
        name="track",
    )


if __name__ == "__main__":
    main()
