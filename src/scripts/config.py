from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
BASE_WEIGHTS = PROJECT_ROOT / "src" / "models" / "yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "src" / "runs"
IMAGE_SIZE = 416
BATCH_SIZE = 8
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda:0")


def select_device():
    """Use an available GPU, otherwise the CPU, without machine-specific paths."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
