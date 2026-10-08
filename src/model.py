from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# On Windows the code must be inside this block, because the data loader
# starts extra processes that import this file again.
if __name__ == "__main__":
    # Load a pre-trained nano model for maximum speed.
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))

    # Train it on our gas cylinder dataset.
    model.train(
        data=str(PROJECT_ROOT / "data" / "data.yaml"),
        project=str(PROJECT_ROOT / "runs"),
        name="train",  # Results go to runs/train.
        exist_ok=True,  # Overwrite runs/train instead of creating train2, train3, ...
        time=0.3,  # Train for 0.3 hours (18 minutes); this replaces epochs.
        fraction=0.2,  # Use only 20% of the training images.
        imgsz=480,  # Image size; must be a multiple of 32.
        batch=8,  # Small batch size to prevent memory errors.
    )
