import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
QUICK_EPOCHS = 3
FULL_EPOCHS = 50
QUICK_FRACTION = 0.1
PATIENCE = 10
IMAGE_SIZE = 416
BATCH_SIZE = 8


def select_device():
    """Prefer the Mac GPU, then CUDA, and otherwise use the CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the gas-cylinder detector.")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Dataset YAML path.")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda:0"), default="auto"
    )
    args = parser.parse_args(argv)

    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset configuration missing: {data_yaml}")

    device = select_device() if args.device == "auto" else args.device
    epochs = QUICK_EPOCHS if args.mode == "quick" else FULL_EPOCHS
    fraction = QUICK_FRACTION if args.mode == "quick" else 1.0
    print(f"Training mode: {args.mode} | Device: {device} | Max epochs: {epochs}")

    # Load a pre-trained nano model for maximum speed.
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    model.train(
        data=str(data_yaml),
        project=str(PROJECT_ROOT / "runs"),
        name=args.mode,
        epochs=epochs,
        patience=PATIENCE,  # Stop after this many epochs without validation improvement.
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        fraction=fraction,
        device=device,
    )
    print(f"Training results: {model.trainer.save_dir}")
    print(f"Best weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
