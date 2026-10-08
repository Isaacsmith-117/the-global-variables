import argparse
from pathlib import Path

from ultralytics import YOLO

if __package__:
    from .model import BATCH_SIZE, DATA_YAML, IMAGE_SIZE, PROJECT_ROOT, select_device
else:
    from model import BATCH_SIZE, DATA_YAML, IMAGE_SIZE, PROJECT_ROOT, select_device


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained gas-cylinder weights.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt.")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Dataset YAML path.")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda:0"), default="auto"
    )
    args = parser.parse_args(argv)

    weights = args.weights.resolve()
    data_yaml = args.data.resolve()
    if not weights.is_file():
        raise SystemExit(f"Trained weights missing: {weights}")
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset configuration missing: {data_yaml}")

    device = select_device() if args.device == "auto" else args.device
    print(f"Evaluation split: {args.split} | Device: {device}")
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split=args.split,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=str(PROJECT_ROOT / "runs"),
        name=f"evaluate_{args.split}",
    )
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Evaluation results: {metrics.save_dir}")


if __name__ == "__main__":
    main()
