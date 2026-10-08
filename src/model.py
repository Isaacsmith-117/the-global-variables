import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
QUICK_HOURS = 0.4  # About 24 minutes, so a CPU run stays under 30 minutes in total.
FULL_EPOCHS = 50
QUICK_FRACTION = 0.2
PATIENCE = 10
IMAGE_SIZE = 480  # Must be a multiple of 32, the model's stride.
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
    fraction = QUICK_FRACTION if args.mode == "quick" else 1.0
    hours = QUICK_HOURS if args.mode == "quick" else None
    print(f"Training mode: {args.mode} | Device: {device}")

    # Load a pre-trained nano model for maximum speed.
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    model.train(
        data=str(data_yaml),
        project=str(PROJECT_ROOT / "runs"),
        name=args.mode,
        epochs=FULL_EPOCHS,
        time=hours,  # Quick mode trains for QUICK_HOURS instead of a number of epochs.
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
