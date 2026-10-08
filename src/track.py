from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Weights from src/model.py. Change "train" to "colab" to use the Colab model.
WEIGHTS = PROJECT_ROOT / "runs" / "train" / "weights" / "best.pt"
VIDEO = PROJECT_ROOT / "data" / "video_banda.mp4"

if __name__ == "__main__":
    model = YOLO(str(WEIGHTS))

    # Detect the cylinders in each frame and give every cylinder an ID
    # that stays the same while it moves on the conveyor belt.
    model.track(
        source=str(VIDEO),
        tracker="bytetrack.yaml",  # Or "botsort.yaml".
        persist=True,  # Keep the IDs from one frame to the next.
        conf=0.4,  # Ignore detections below 40% confidence.
        vid_stride=3,  # Process 1 frame out of 3.
        save=True,  # Save the video with boxes and IDs.
        project=str(PROJECT_ROOT / "runs"),
        name="track",  # Results go to runs/track.
        exist_ok=True,
    )
