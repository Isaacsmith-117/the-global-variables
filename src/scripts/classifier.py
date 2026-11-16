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
    from .reason_schema import LEGACY_REASONS, PROJECT_POLICY, REASONS, read_reason_schema
else:
    from config import DEVICE_CHOICES, select_device
    from reason_schema import LEGACY_REASONS, PROJECT_POLICY, REASONS, read_reason_schema

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

    def __init__(self, pretrained=False, reason_order=REASONS):
        super().__init__()
        self.reason_order = tuple(reason_order)
        if self.reason_order not in (REASONS, LEGACY_REASONS):
            raise ValueError("Unsupported reason order.")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        feature_count = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(feature_count, len(self.reason_order))
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Frozen BatchNorm statistics must stay frozen too.
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)


def load_dataset(root, allow_legacy=False):
    """Validate the prepared manifest before any images are used for training."""
    root = Path(root)
    metadata = json.loads((root / "dataset.json").read_text())
    version, reason_order, _ = read_reason_schema(metadata, allow_legacy)
    enabled = metadata["enabled_reason_outputs"]
    if len(set(enabled)) != len(enabled) or not set(enabled) <= set(reason_order):
        raise ValueError("Enabled reasons must be a subset of the dataset reason order.")
    rows = [json.loads(line) for line in (root / metadata["manifest"]).read_text().splitlines()
            if line.strip()]
    groups, crops, hashes, views_by_group = {}, set(), {}, {}
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
        image_hash = row.get("sha256")
        if image_hash and hashes.setdefault(image_hash, split) != split:
            raise ValueError("The same image appears in multiple splits.")
        if (len(row["reason_targets"]) != len(reason_order)
                or len(row["reason_loss_mask"]) != len(reason_order)):
            raise ValueError(f"Expected {len(reason_order)} reason targets and masks.")
        for target, mask in zip(row["reason_targets"], row["reason_loss_mask"]):
            if target not in (None, 0, 1) or mask not in (0, 1) or (mask and target is None):
                raise ValueError("Unknown targets must be masked out; targets must be 0/1/null.")
        if (split != "quarantine" and any(row["reason_loss_mask"])
                and (row.get("quarantined") or row.get("training_approved") is not True)):
            raise ValueError(f"Unapproved crop outside quarantine: {row['crop_id']}")
        if version == 2 and split != "quarantine" and any(row["reason_loss_mask"]):
            valid_source = (row.get("source") == "video" and row.get("camera_id") in CAMERAS
                            or row.get("source") == "instructor_photo" and row.get("camera_id") in (None, ""))
            if not valid_source:
                raise ValueError(f"Known labels need a primary video camera or instructor photo: {row['crop_id']}")
    if version == 1:
        # The historical release contains fixed triplets, unlike v2 per-image labels.
        for group, views in views_by_group.items():
            if len(views) != 3 or {r["camera_id"] for r in views} != set(CAMERAS):
                raise ValueError(f"Expected one crop per camera for {group}.")
            statuses = {r["primagaz_status"] for r in views}
            if len(statuses) != 1 or not statuses <= {"OK", "NOK"}:
                raise ValueError(f"Inconsistent plant status for {group}.")
    return metadata, rows


def targets_and_masks(rows, enabled, reason_order=REASONS):
    targets = torch.tensor([[0 if v is None else v for v in r["reason_targets"]]
                            for r in rows], dtype=torch.float32)
    masks = torch.tensor([r["reason_loss_mask"] for r in rows], dtype=torch.float32)
    masks *= torch.tensor([reason in enabled for reason in reason_order])
    return targets, masks


def masked_loss(logits, targets, masks, pos_weight=None):
    """Unknown/disabled labels contribute neither loss nor gradient."""
    if masks.sum() == 0:
        raise ValueError("No known enabled labels in this batch.")
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none")
    return (losses * masks).sum() / masks.sum()


def primary_video_triplets(rows, legacy=False):
    """Select complete primary-video triplets, never arbitrary sets of photos."""
    groups = {}
    for row in rows:
        if (not row.get("quarantined") and not row.get("calibration_excluded")
                and row["split"] != "quarantine"
                and (row.get("source") == "video" or legacy and "source" not in row)
                and (legacy or row.get("primary_view", True) in (True, 1, "1", "true"))
                and row.get("camera_id") in CAMERAS):
            groups.setdefault(row["group_id"], []).append(row)
    return [row for views in groups.values()
            if len(views) == 3 and {r["camera_id"] for r in views} == set(CAMERAS)
            for row in views]


def bottle_scores(rows, crop_scores, reason_order=REASONS):
    """Aggregate fixed triplets; a negative truth requires three known negatives."""
    crop_scores = torch.as_tensor(crop_scores, dtype=torch.float32)
    if (crop_scores.shape != (len(rows), len(reason_order))
            or not torch.isfinite(crop_scores).all()
            or ((crop_scores < 0) | (crop_scores > 1)).any()):
        raise ValueError("Expected one finite probability score per image and reason.")
    bottles = {}
    for i, row in enumerate(rows):
        bottles.setdefault(row["group_id"], []).append(i)
    results = []
    for group, indices in bottles.items():
        views = [rows[i] for i in indices]
        if len(views) != 3 or {r["camera_id"] for r in views} != set(CAMERAS):
            raise ValueError(f"Incomplete camera triplet: {group}")
        truth = []
        for j in range(len(reason_order)):
            known = [r["reason_targets"][j] if r["reason_loss_mask"][j] else None
                     for r in views]
            truth.append(1 if 1 in known else 0 if all(v == 0 for v in known) else None)
        statuses = {r.get("primagaz_status") for r in views}
        plant_status = next(iter(statuses)) if len(statuses) == 1 else None
        results.append({"bottle_id": group, "plant_status": plant_status,
                        "scores": crop_scores[indices].amax(dim=0).tolist(), "truth": truth})
    return results


def decide(scores, thresholds, enabled, reason_order=REASONS):
    """One final decision from three per-view reason scores in CAMERAS order."""
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.shape != (3, len(reason_order)) or not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected finite scores between 0 and 1 for three views/reasons.")
    if not enabled or len(set(enabled)) != len(enabled) or not set(enabled) <= set(reason_order):
        raise ValueError("No valid enabled reasons.")
    reasons, evidence, combined = [], {}, {}
    for reason in enabled:
        threshold = thresholds.get(reason)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError(f"Missing or invalid validation threshold for {reason}.")
        j = reason_order.index(reason)
        combined[reason] = scores[:, j].max().item()
        cameras = [camera for i, camera in enumerate(CAMERAS) if scores[i, j] >= threshold]
        if cameras:
            reasons.append(reason)
            evidence[reason] = cameras
    return {"status": "NOK" if reasons else "OK", "reasons": reasons,
            "reason_scores": combined, "evidence_views": evidence,
            "checked_reasons": list(enabled),
            "unchecked_reasons": [r for r in reason_order if r not in enabled]}


class BottleClassifier:
    """Load once, then call predict with three matched RGB crops or image paths."""

    def __init__(self, checkpoint, device="cpu", demo_dirt=False, binary_checkpoint=None,
                 binary_aggregation="max", nok_threshold=None):
        if binary_aggregation not in ("max", "mean"):
            raise ValueError("Binary aggregation must be max or mean across all three views.")
        if binary_checkpoint is None and (nok_threshold is not None or binary_aggregation != "max"):
            raise ValueError("Binary settings require a binary checkpoint.")
        if binary_checkpoint is not None and demo_dirt:
            raise ValueError("Binary inference cannot use the untrained --demo-dirt output.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.reason_schema_version, self.reason_order, self.project_policy = read_reason_schema(
            state, allow_legacy=True)
        if state["image_size"] != IMAGE_SIZE:
            raise ValueError("Checkpoint format does not match this classifier.")
        if demo_dirt and self.reason_schema_version != 1:
            raise ValueError("--demo-dirt is legacy-only; v2 contamination needs reviewed training labels.")
        self.model = ReasonModel(reason_order=self.reason_order)
        self.model.load_state_dict(state["model_state"])
        self.binary_head = None
        self.binary_threshold = None
        self.binary_project_policy = None
        self.binary_policy_unverified = False
        self.binary_aggregation = binary_aggregation
        if binary_checkpoint is not None:
            binary = torch.load(binary_checkpoint, map_location="cpu", weights_only=True)
            supported = (
                binary.get("format") == "instructor_visual_ok_nok_v1"
                and binary.get("backbone") == "mobilenet_v3_small"
                and binary.get("positive_label") == "nok"
            ) or (binary.get("format") == "gass_binary_experiment_v1"
                  and binary.get("model") == "mobilenet")
            if not supported or binary.get("image_size") != IMAGE_SIZE:
                raise ValueError("Expected a supported MobileNet binary checkpoint with image size 224.")
            binary_policy = binary.get("project_policy")
            if binary_policy is not None and binary_policy != PROJECT_POLICY:
                raise ValueError("Binary checkpoint project policy conflicts with primagaz_only_v1.")
            self.binary_project_policy = binary_policy
            self.binary_policy_unverified = binary_policy is None
            threshold = binary.get("threshold") if nok_threshold is None else nok_threshold
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise ValueError("NOK threshold must be finite and between 0 and 1.")
            weights = binary["state_dict"]
            expected = self.model.state_dict()
            expected_encoder = {k: v for k, v in expected.items() if k.startswith("backbone.")}
            binary_encoder = {k: v for k, v in weights.items() if k.startswith("backbone.")}
            if set(binary_encoder) != set(expected_encoder) or any(
                not torch.equal(value, binary_encoder[key]) for key, value in expected_encoder.items()):
                raise ValueError("Binary and reason checkpoints must have identical frozen encoders.")
            head = {key.removeprefix("head."): value for key, value in weights.items()
                    if key.startswith("head.")}
            if any(not torch.isfinite(value).all() for value in head.values()):
                raise ValueError("Binary head contains nonfinite weights.")
            self.binary_head = nn.Linear(self.model.head.in_features, 1)
            self.binary_head.load_state_dict(head)
            self.binary_head.to(device).eval()
            self.binary_threshold = threshold
        self.device = device
        self.model.to(device).eval()
        self.thresholds = dict(state["thresholds"])
        self.enabled = list(state["enabled_reasons"])
        self.demo_only_reasons = []
        if demo_dirt and "heavy_dirt" not in self.enabled:
            # Explicit demo opt-in; this is not a learned or calibrated dirt output.
            self.enabled = [r for r in self.reason_order if r in self.enabled or r == "heavy_dirt"]
            self.thresholds["heavy_dirt"] = 0.5
            self.demo_only_reasons = ["heavy_dirt"]
        decide(torch.zeros(3, len(self.reason_order)), self.thresholds, self.enabled, self.reason_order)

    @torch.inference_mode()
    def predict(self, crops, bottle_id=None):
        if set(crops) != set(CAMERAS) or any(v is None for v in crops.values()):
            raise ValueError("Provide all three matched views; an incomplete inspection is not OK.")
        images = [preprocess(crops[c]) if isinstance(crops[c], Image.Image)
                  else read_crop(crops[c]) for c in CAMERAS]
        # Both output layers use the same frozen image features.
        features = self.model.backbone(torch.stack(images).to(self.device))
        scores = self.model.head(features).sigmoid().cpu()
        result = {"bottle_id": bottle_id,
                  **decide(scores, self.thresholds, self.enabled, self.reason_order),
                  "reason_schema_version": self.reason_schema_version,
                  "legacy_reason_schema": self.reason_schema_version == 1,
                  "reason_order": list(self.reason_order), "project_policy": self.project_policy,
                  "demo_only_reasons": self.demo_only_reasons}
        if self.binary_head is not None:
            binary_scores = self.binary_head(features).sigmoid().squeeze(1).cpu()
            if not torch.isfinite(binary_scores).all():
                raise ValueError("Binary classifier returned nonfinite scores.")
            view_scores = dict(zip(CAMERAS, binary_scores.tolist()))
            score = max(view_scores.values()) if self.binary_aggregation == "max" else sum(view_scores.values()) / 3
            status = "NOK" if score >= self.binary_threshold else "OK"
            reason_status = result["status"]
            result.update(reason_status=reason_status, status=status,
                          decision_source=f"binary_{self.binary_aggregation}", binary_scores=view_scores,
                          binary_score=score, binary_aggregation=self.binary_aggregation,
                          binary_threshold=self.binary_threshold,
                          binary_project_policy=self.binary_project_policy,
                          binary_policy_unverified=self.binary_policy_unverified,
                          reason_unresolved=status == "NOK" and not result["reasons"],
                          reason_disagreement=status != reason_status)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify one matched three-view bottle.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--binary-checkpoint", type=Path,
                        help="Optional MobileNet binary checkpoint; all three views decide, reasons stay separate.")
    parser.add_argument("--binary-aggregation", choices=("max", "mean"),
                        help="Combine all three NOK scores by maximum (default) or arithmetic mean.")
    parser.add_argument("--nok-threshold", type=float,
                        help="NOK cutoff in [0, 1]; defaults to the binary checkpoint's saved cutoff.")
    for camera in CAMERAS:
        parser.add_argument("--" + camera.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--bottle-id", default=None)
    parser.add_argument("--demo-dirt", action="store_true",
                        help="Include the untrained dirt output at a 0.5 cutoff for a demo.")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args(argv)
    if (args.binary_aggregation is not None or args.nok_threshold is not None) and not args.binary_checkpoint:
        parser.error("--binary-aggregation and --nok-threshold require --binary-checkpoint.")
    if args.nok_threshold is not None and not 0 <= args.nok_threshold <= 1:
        parser.error("--nok-threshold must be finite and between 0 and 1.")
    if args.binary_checkpoint and args.demo_dirt:
        parser.error("--binary-checkpoint cannot be combined with --demo-dirt.")
    device = select_device() if args.device == "auto" else args.device
    classifier = BottleClassifier(args.checkpoint, device, demo_dirt=args.demo_dirt,
                                  binary_checkpoint=args.binary_checkpoint,
                                  binary_aggregation=args.binary_aggregation or "max",
                                  nok_threshold=args.nok_threshold)
    result = classifier.predict({c: getattr(args, c) for c in CAMERAS}, args.bottle_id)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
