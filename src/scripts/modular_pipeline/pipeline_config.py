"""Standalone configuration for the modular three-camera pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / "src" / "runs"

DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda:0")
IMAGE_SIZE = 416

VIDEO_CAMERAS = ("back_left", "back_right", "front")
CAMERA_INSPECTION_ORDER = ("front", "back_left", "back_right")
DEFAULT_VIDEOS = [
    PROJECT_ROOT / "data" / "primagaz_vids" / "14_55" / f"14_55_{name}_cropped.mp4"
    for name in VIDEO_CAMERAS
]

MIN_TRACK_FRAMES = 40
MIN_MOVE_FRAC = 0.4
LINE_FRAC = 0.5
TILE_WIDTH = 640
MAX_CROSSING_SPREAD_FRAMES = 15

TRACK_CONFIDENCE = 0.1
TRACKER = "bytetrack.yaml"

GREY = (160, 160, 160)
COLORS = (
    (0, 220, 80),
    (255, 160, 0),
    (255, 60, 200),
    (0, 200, 255),
    (80, 80, 255),
    (200, 255, 0),
    (255, 255, 255),
    (160, 100, 255),
)

REASON_SCHEMA_VERSION = 2
REASONS = ("corrosion", "mechanical_damage", "heavy_contamination", "wrong_brand_or_type")
LEGACY_REASONS = ("major_damage", "heavy_dirt", "wrong_brand_or_model")
PROJECT_POLICY = "primagaz_only_v1"
REASON_LABELS = {
    "corrosion": "Significant corrosion",
    "mechanical_damage": "Major mechanical damage",
    "heavy_contamination": "Heavy contamination",
    "wrong_brand_or_type": "Wrong brand or bottle type",
}

OCR_FIELDS = ("tare_kg", "approval_until_date")
TARE_TOLERANCE_KG = 0.1
OCR_SCALE = 2
OCR_ENLARGE_BELOW = 1000
OCR_PAD = 4
OCR_PROPOSAL_CONFIDENCE = 0.01
OCR_YELLOW = (255, 220, 0)
OCR_GREEN = (0, 200, 0)
OCR_MIN_DATE_YEAR = 2000

CLASSIFIER_IMAGE_SIZE = 224
CLASSIFIER_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
CLASSIFIER_NORMALIZE_STD = (0.229, 0.224, 0.225)
DEMO_DIRT_THRESHOLD = 0.5
