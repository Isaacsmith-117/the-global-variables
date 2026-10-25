"""Shared crop model and three-view bottle decision for the WS5 visual checks."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms.functional import normalize, pil_to_tensor

if __package__:
    from .config import DEVICE_CHOICES, select_device
else:
    from config import DEVICE_CHOICES, select_device

REASONS = ("major_damage", "heavy_dirt", "wrong_brand_or_model")
CAMERAS = ("front", "back_left", "back_right")
IMAGE_SIZE = 224


def preprocess(image):
    """Preserve the whole RGB bottle, pad to a square, then normalize."""
    image = ImageOps.exif_transpose(image)
    image = ImageOps.pad(image.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE),
                         method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    tensor = pil_to_tensor(image).float() / 255
    return normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def read_crop(path):
    with Image.open(path) as image:
        return preprocess(image)


class ReasonModel(nn.Module):
    """Frozen MobileNet features with one trainable linear reason layer."""

    def __init__(self, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        feature_count = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(feature_count, len(REASONS))
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Frozen BatchNorm statistics must stay frozen too.
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)


def load_dataset(root):
    """Validate the prepared manifest before any images are used for training."""
    root = Path(root)
    metadata = json.loads((root / "dataset.json").read_text())
    if metadata["reason_order"] != list(REASONS):
        raise ValueError("Dataset reason order does not match the classifier.")
    enabled = metadata["enabled_reason_outputs"]
    if not enabled or len(set(enabled)) != len(enabled) or not set(enabled) <= set(REASONS):
        raise ValueError("Enabled reasons must be a nonempty subset of REASONS.")
    rows = [json.loads(line) for line in (root / metadata["manifest"]).read_text().splitlines()
            if line.strip()]
    groups, crops, views_by_group = {}, set(), {}
    for row in rows:
        split, group = row["split"], row["group_id"]
        if split not in ("train", "validation", "test", "quarantine"):
            raise ValueError(f"Invalid split: {split}")
        if group in groups and groups[group] != split:
            raise ValueError(f"Bottle appears in multiple splits: {group}")
        groups[group] = split
        views_by_group.setdefault(group, []).append(row)
        if row["crop_id"] in crops:
            raise ValueError(f"Duplicate crop: {row['crop_id']}")
        crops.add(row["crop_id"])
        if len(row["reason_targets"]) != 3 or len(row["reason_loss_mask"]) != 3:
            raise ValueError("Expected three reason targets and masks.")
        for target, mask in zip(row["reason_targets"], row["reason_loss_mask"]):
            if target not in (None, 0, 1) or mask not in (0, 1) or (mask and target is None):
                raise ValueError("Unknown targets must be masked out; targets must be 0/1/null.")
        if split != "quarantine" and not row["training_approved"]:
            raise ValueError(f"Unapproved crop outside quarantine: {row['crop_id']}")
    # Every eligible bottle must have exactly one crop from each camera.
    for group, views in views_by_group.items():
        if len(views) != 3 or {r["camera_id"] for r in views} != set(CAMERAS):
            raise ValueError(f"Expected one crop per camera for {group}.")
        statuses = {r["primagaz_status"] for r in views}
        if len(statuses) != 1 or not statuses <= {"OK", "NOK"}:
            raise ValueError(f"Inconsistent plant status for {group}.")
    return metadata, rows


def targets_and_masks(rows, enabled):
    targets = torch.tensor([[0 if v is None else v for v in r["reason_targets"]]
                            for r in rows], dtype=torch.float32)
    masks = torch.tensor([r["reason_loss_mask"] for r in rows], dtype=torch.float32)
    masks *= torch.tensor([reason in enabled for reason in REASONS])
    return targets, masks


def masked_loss(logits, targets, masks, pos_weight=None):
    """Unknown/disabled labels contribute neither loss nor gradient."""
    if masks.sum() == 0:
        raise ValueError("No known enabled labels in this batch.")
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none")
    return (losses * masks).sum() / masks.sum()


def bottle_scores(rows, crop_scores):
    """Aggregate fixed triplets; a negative truth requires three known negatives."""
    bottles = {}
    for i, row in enumerate(rows):
        bottles.setdefault(row["group_id"], []).append(i)
    results = []
    for group, indices in bottles.items():
        views = [rows[i] for i in indices]
        if len(views) != 3 or {r["camera_id"] for r in views} != set(CAMERAS):
            raise ValueError(f"Incomplete camera triplet: {group}")
        truth = []
        for j in range(len(REASONS)):
            known = [r["reason_targets"][j] if r["reason_loss_mask"][j] else None
                     for r in views]
            truth.append(1 if 1 in known else 0 if all(v == 0 for v in known) else None)
        results.append({"bottle_id": group, "plant_status": views[0]["primagaz_status"],
                        "scores": crop_scores[indices].amax(dim=0).tolist(), "truth": truth})
    return results


def decide(scores, thresholds, enabled):
    """One final decision from three per-view reason scores in CAMERAS order."""
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.shape != (3, 3) or not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected finite scores between 0 and 1 for three views/reasons.")
    if not enabled or not set(enabled) <= set(REASONS):
        raise ValueError("No valid enabled reasons.")
    reasons, evidence, combined = [], {}, {}
    for reason in enabled:
        threshold = thresholds.get(reason)
        if threshold is None or not 0 < threshold <= 1:
            raise ValueError(f"Missing or invalid validation threshold for {reason}.")
        j = REASONS.index(reason)
        combined[reason] = scores[:, j].max().item()
        cameras = [camera for i, camera in enumerate(CAMERAS) if scores[i, j] >= threshold]
        if cameras:
            reasons.append(reason)
            evidence[reason] = cameras
    return {"status": "NOK" if reasons else "OK", "reasons": reasons,
            "reason_scores": combined, "evidence_views": evidence,
            "checked_reasons": list(enabled),
            "unchecked_reasons": [r for r in REASONS if r not in enabled]}


class BottleClassifier:
    """Load once, then call predict with three matched RGB crops or image paths."""

    def __init__(self, checkpoint, device="cpu", demo_dirt=False):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["reason_order"] != list(REASONS) or state["image_size"] != IMAGE_SIZE:
            raise ValueError("Checkpoint format does not match this classifier.")
        self.model = ReasonModel()
        self.model.load_state_dict(state["model_state"])
        self.device = device
        self.model.to(device).eval()
        self.thresholds = dict(state["thresholds"])
        self.enabled = list(state["enabled_reasons"])
        self.demo_only_reasons = []
        if demo_dirt and "heavy_dirt" not in self.enabled:
            # Explicit demo opt-in; this is not a learned or calibrated dirt output.
            self.enabled = [r for r in REASONS if r in self.enabled or r == "heavy_dirt"]
            self.thresholds["heavy_dirt"] = 0.5
            self.demo_only_reasons = ["heavy_dirt"]
        decide(torch.zeros(3, 3), self.thresholds, self.enabled)

    @torch.inference_mode()
    def predict(self, crops, bottle_id=None):
        if set(crops) != set(CAMERAS) or any(v is None for v in crops.values()):
            raise ValueError("Provide all three matched views; an incomplete inspection is not OK.")
        images = [preprocess(crops[c]) if isinstance(crops[c], Image.Image)
                  else read_crop(crops[c]) for c in CAMERAS]
        scores = self.model(torch.stack(images).to(self.device)).sigmoid().cpu()
        return {"bottle_id": bottle_id, **decide(scores, self.thresholds, self.enabled),
                "demo_only_reasons": self.demo_only_reasons}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify one matched three-view bottle.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    for camera in CAMERAS:
        parser.add_argument("--" + camera.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--bottle-id", default=None)
    parser.add_argument("--demo-dirt", action="store_true",
                        help="Include the untrained dirt output at a 0.5 cutoff for a demo.")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args(argv)
    device = select_device() if args.device == "auto" else args.device
    classifier = BottleClassifier(args.checkpoint, device, demo_dirt=args.demo_dirt)
    result = classifier.predict({c: getattr(args, c) for c in CAMERAS}, args.bottle_id)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
