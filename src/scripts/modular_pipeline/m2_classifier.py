"""Standalone visual classifier for the modular three-camera pipeline."""

import argparse
import json
import torch
from pathlib import Path

from PIL import Image, ImageOps
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms.functional import normalize, pil_to_tensor

try:
    from .pipeline_config import (
        CAMERA_INSPECTION_ORDER, CLASSIFIER_IMAGE_SIZE,
        CLASSIFIER_NORMALIZE_MEAN, CLASSIFIER_NORMALIZE_STD,
        DEMO_DIRT_THRESHOLD, LEGACY_REASONS, PROJECT_POLICY, REASONS,
    )
except ImportError:
    from pipeline_config import (
        CAMERA_INSPECTION_ORDER, CLASSIFIER_IMAGE_SIZE,
        CLASSIFIER_NORMALIZE_MEAN, CLASSIFIER_NORMALIZE_STD,
        DEMO_DIRT_THRESHOLD, LEGACY_REASONS, PROJECT_POLICY, REASONS,
    )

CAMERAS = CAMERA_INSPECTION_ORDER
IMAGE_SIZE = CLASSIFIER_IMAGE_SIZE


def read_reason_schema(metadata, allow_legacy=False):
    version = metadata.get("reason_schema_version")
    raw_order = metadata.get("reason_order")
    order = tuple(raw_order) if isinstance(raw_order, (list, tuple)) else ()
    if (type(version) is int and version == 2
            and order == REASONS and metadata.get("project_policy") == PROJECT_POLICY):
        return 2, order, PROJECT_POLICY
    if (allow_legacy and (version is None or type(version) is int and version == 1)
            and order == LEGACY_REASONS and metadata.get("project_policy") is None):
        return 1, order, None
    raise ValueError("Expected schema 2 or an explicitly supported legacy schema 1 checkpoint.")


def preprocess(image):
    image = ImageOps.exif_transpose(image)
    image = ImageOps.pad(image.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE),
                         method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    tensor = pil_to_tensor(image).float() / 255
    return normalize(tensor, CLASSIFIER_NORMALIZE_MEAN, CLASSIFIER_NORMALIZE_STD)


def read_crop(path):
    with Image.open(path) as image:
        return preprocess(image)


class ReasonModel(nn.Module):
    """Frozen MobileNet features with one linear reason head."""

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
        self.backbone.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)


def decide(scores, thresholds, enabled, reason_order=REASONS):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.shape != (3, len(reason_order)) or not torch.isfinite(scores).all():
        raise ValueError("Expected finite scores for three views and reasons.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected scores between 0 and 1.")
    if not enabled or len(set(enabled)) != len(enabled) or not set(enabled) <= set(reason_order):
        raise ValueError("No valid enabled reasons.")
    reasons, evidence, combined = [], {}, {}
    for reason in enabled:
        threshold = thresholds.get(reason)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError("Missing or invalid validation threshold for reason.")
        index = reason_order.index(reason)
        combined[reason] = scores[:, index].max().item()
        cameras = [camera for camera_index, camera in enumerate(CAMERAS)
                   if scores[camera_index, index] >= threshold]
        if cameras:
            reasons.append(reason)
            evidence[reason] = cameras
    return {"status": "NOK" if reasons else "OK", "reasons": reasons,
            "reason_scores": combined, "evidence_views": evidence,
            "checked_reasons": list(enabled),
            "unchecked_reasons": [reason for reason in reason_order if reason not in enabled]}


class BottleClassifier:
    """Load once, then classify three matched RGB crops or image paths."""

    def __init__(self, checkpoint, device="cpu", demo_dirt=False, binary_checkpoint=None,
                 binary_aggregation="max", nok_threshold=None):
        if binary_aggregation not in ("max", "mean"):
            raise ValueError("Binary aggregation must be max or mean across all three views.")
        if binary_checkpoint is None and (nok_threshold is not None or binary_aggregation != "max"):
            raise ValueError("Binary settings require a binary checkpoint.")
        if binary_checkpoint is not None and demo_dirt:
            raise ValueError("Binary inference cannot use the untrained demo dirt output.")

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
            supported = ((binary.get("format") == "instructor_visual_ok_nok_v1"
                          and binary.get("backbone") == "mobilenet_v3_small"
                          and binary.get("positive_label") == "nok")
                         or (binary.get("format") == "gass_binary_experiment_v1"
                             and binary.get("model") == "mobilenet"))
            if not supported or binary.get("image_size") != IMAGE_SIZE:
                raise ValueError(f"Expected a supported MobileNet binary checkpoint with image size {IMAGE_SIZE}.")
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
            expected_encoder = {key: value for key, value in expected.items() if key.startswith("backbone.")}
            binary_encoder = {key: value for key, value in weights.items() if key.startswith("backbone.")}
            if set(binary_encoder) != set(expected_encoder) or any(
                not torch.equal(value, binary_encoder[key]) for key, value in expected_encoder.items()
            ):
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
            self.enabled = [reason for reason in self.reason_order
                            if reason in self.enabled or reason == "heavy_dirt"]
            self.thresholds["heavy_dirt"] = DEMO_DIRT_THRESHOLD
            self.demo_only_reasons = ["heavy_dirt"]
        decide(torch.zeros(3, len(self.reason_order)), self.thresholds,
               self.enabled, self.reason_order)

    @torch.inference_mode()
    def predict(self, crops, bottle_id=None):
        if set(crops) != set(CAMERAS) or any(view is None for view in crops.values()):
            raise ValueError("Provide all three matched views; an incomplete inspection is not OK.")
        images = [preprocess(crops[camera]) if isinstance(crops[camera], Image.Image)
                  else read_crop(crops[camera]) for camera in CAMERAS]
        features = self.model.backbone(torch.stack(images).to(self.device))
        scores = self.model.head(features).sigmoid().cpu()
        result = {"bottle_id": bottle_id,
                  **decide(scores, self.thresholds, self.enabled, self.reason_order),
                  "reason_schema_version": self.reason_schema_version,
                  "legacy_reason_schema": self.reason_schema_version == 1,
                  "reason_order": list(self.reason_order),
                  "project_policy": self.project_policy,
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
                          decision_source=f"binary_{self.binary_aggregation}",
                          binary_scores=view_scores, binary_score=score,
                          binary_aggregation=self.binary_aggregation,
                          binary_threshold=self.binary_threshold,
                          binary_project_policy=self.binary_project_policy,
                          binary_policy_unverified=self.binary_policy_unverified,
                          reason_unresolved=status == "NOK" and not result["reasons"],
                          reason_disagreement=status != reason_status)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify one matched three-view bottle.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--binary-checkpoint", type=Path)
    parser.add_argument("--binary-aggregation", choices=("max", "mean"))
    parser.add_argument("--nok-threshold", type=float)
    for camera in CAMERAS:
        parser.add_argument("--" + camera.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--bottle-id", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    classifier = BottleClassifier(args.checkpoint, args.device,
                                  binary_checkpoint=args.binary_checkpoint,
                                  binary_aggregation=args.binary_aggregation or "max",
                                  nok_threshold=args.nok_threshold)
    result = classifier.predict({camera: getattr(args, camera) for camera in CAMERAS}, args.bottle_id)
    print(json.dumps(result, indent=2))
    return result


__all__ = ["BottleClassifier", "decide", "preprocess"]
