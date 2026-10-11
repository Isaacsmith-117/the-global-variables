import argparse
from pathlib import Path

from ultralytics import YOLO

if __package__:
    from .config import BASE_WEIGHTS, BATCH_SIZE, DATA_YAML, DEVICE_CHOICES, IMAGE_SIZE, RUNS_DIR, select_device
else:
    from config import BASE_WEIGHTS, BATCH_SIZE, DATA_YAML, DEVICE_CHOICES, IMAGE_SIZE, RUNS_DIR, select_device

QUICK_EPOCHS = 3
FULL_EPOCHS = 50
QUICK_FRACTION = 0.1
PATIENCE = 10


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the gas-cylinder detector.")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Dataset YAML path.")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args(argv)

    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset configuration missing: {data_yaml}")

    device = select_device() if args.device == "auto" else args.device
    epochs = QUICK_EPOCHS if args.mode == "quick" else FULL_EPOCHS
    fraction = QUICK_FRACTION if args.mode == "quick" else 1.0
    print(f"Training mode: {args.mode} | Device: {device} | Max epochs: {epochs}")

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(data_yaml),
        project=str(RUNS_DIR),
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
