from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Weights from src/model.py. Change "train" to "colab" to use the Colab model.
WEIGHTS = PROJECT_ROOT / "runs" / "train" / "weights" / "best.pt"

# On Windows the code must be inside this block, because the data loader
# starts extra processes that import this file again.
if __name__ == "__main__":
    model = YOLO(str(WEIGHTS))

    # Test the model on the test images, which it never saw during training.
    # Ultralytics prints precision, recall, mAP50 and mAP50-95.
    model.val(
        data=str(PROJECT_ROOT / "data" / "data.yaml"),
        split="test",
        project=str(PROJECT_ROOT / "runs"),
        name="evaluate",  # Results go to runs/evaluate.
        exist_ok=True,
    )
