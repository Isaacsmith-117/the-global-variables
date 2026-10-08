from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    data_yaml = PROJECT_ROOT / "data" / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset configuration missing: {data_yaml}")

    # Load a pre-trained nano model for maximum speed.
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    model.train(
        data=str(data_yaml),
        project=str(PROJECT_ROOT / "runs"),
        epochs=3,  # Just 3 epochs to see if it runs.
        patience=10,  # Stop after 10 epochs without validation improvement.
        imgsz=416,  # Lower resolution speeds up the test.
        batch=8,  # Small batch size to prevent memory errors.
        fraction=0.1,  # Uses only 10% of the training dataset.
    )


if __name__ == "__main__":
    main()
