import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.scripts import classifier as c
from src.scripts import train_classifier as training


def triplet(group, split, targets=(0, 0, 0), masks=(1, 0, 1)):
    return [{"group_id": group, "crop_id": f"{group}_{camera}", "camera_id": camera,
             "image_path": f"{group}_{camera}.jpg", "split": split,
             "training_approved": True, "source": "video", "primagaz_status": "NOK" if 1 in targets else "OK",
             "reason_targets": list(targets), "reason_loss_mask": list(masks)}
            for camera in c.CAMERAS]


def write_dataset(root, rows, legacy=False, enabled=None):
    order = c.LEGACY_REASONS if legacy else c.REASONS
    metadata = {"manifest": "manifest.jsonl", "reason_order": list(order),
                "enabled_reason_outputs": enabled if enabled is not None else [order[0], order[-1]],
                "dataset_version": "synthetic"}
    if not legacy:
        metadata.update(reason_schema_version=2, project_policy=c.PROJECT_POLICY)
    (root / "dataset.json").write_text(json.dumps(metadata))
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def inference_checkpoints(root, legacy=True):
    """Matching synthetic encoder weights; no downloaded weights or private images."""
    order = c.LEGACY_REASONS if legacy else c.REASONS
    model = c.ReasonModel(reason_order=order)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-4)
        model.head.bias[0] = 4
    reason_path, binary_path = root / "reason.pt", root / "binary.pt"
    reason = {"model_state": model.state_dict(), "reason_order": list(order),
              "image_size": c.IMAGE_SIZE, "enabled_reasons": [order[0]],
              "thresholds": {order[0]: 0.5}}
    if not legacy:
        reason.update(reason_schema_version=2, project_policy=c.PROJECT_POLICY)
    torch.save(reason, reason_path)
    weights = dict(model.state_dict())
    weights["head.weight"] = torch.zeros(1, model.head.in_features)
    weights["head.weight"][0, 0] = 1
    weights["head.bias"] = torch.zeros(1)
    binary = {"format": "instructor_visual_ok_nok_v1", "backbone": "mobilenet_v3_small",
              "positive_label": "nok", "image_size": c.IMAGE_SIZE,
              "threshold": 0.5, "state_dict": weights}
    torch.save(binary, binary_path)
    return reason_path, binary_path, binary


class ClassifierTests(unittest.TestCase):
    def test_v2_four_reasons_share_encoder_with_binary_and_keep_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            reason_path, binary_path, _ = inference_checkpoints(Path(directory), legacy=False)
            loaded = c.BottleClassifier(reason_path, binary_checkpoint=binary_path)
            self.assertEqual(loaded.model.head.out_features, 4)
            self.assertEqual(loaded.binary_head.out_features, 1)
            loaded.enabled = list(c.REASONS)
            loaded.thresholds = dict.fromkeys(c.REASONS, 0.5)
            with torch.no_grad():
                loaded.model.head.bias.fill_(4)
            crops = {camera: Image.new("RGB", (20, 40)) for camera in c.CAMERAS}
            features = torch.zeros(3, loaded.model.head.in_features)
            with patch.object(loaded.model.backbone, "forward", return_value=features) as encoder:
                result = loaded.predict(crops)
            encoder.assert_called_once()
            self.assertEqual(result["reasons"], list(c.REASONS))
            self.assertEqual(result["reason_schema_version"], 2)
            self.assertFalse(result["legacy_reason_schema"])
            self.assertEqual(result["reason_order"], list(c.REASONS))
            self.assertEqual(result["project_policy"], "primagaz_only_v1")
            self.assertEqual(set(result["evidence_views"]), set(c.REASONS))
            self.assertEqual(set(result["binary_scores"]), set(c.CAMERAS))
            self.assertIsNone(result["binary_project_policy"])
            self.assertTrue(result["binary_policy_unverified"])

    def test_binary_project_policy_is_separate_from_reason_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            reason_path, binary_path, binary = inference_checkpoints(Path(directory), legacy=False)
            for policy in (None, c.PROJECT_POLICY):
                torch.save({**binary, "project_policy": policy}, binary_path)
                loaded = c.BottleClassifier(reason_path, binary_checkpoint=binary_path)
                self.assertEqual(loaded.binary_project_policy, policy)
                self.assertEqual(loaded.binary_policy_unverified, policy is None)
            torch.save({**binary, "project_policy": "primagaz_and_benegas"}, binary_path)
            with self.assertRaisesRegex(ValueError, "Binary checkpoint project policy conflicts"):
                c.BottleClassifier(reason_path, binary_checkpoint=binary_path)

    def test_legacy_checkpoint_names_and_demo_dirt_remain_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            reason_path, _, _ = inference_checkpoints(Path(directory))
            loaded = c.BottleClassifier(reason_path)
            demo = c.BottleClassifier(reason_path, demo_dirt=True)
            self.assertEqual(loaded.model.head.out_features, 3)
            self.assertEqual(loaded.reason_schema_version, 1)
            self.assertEqual(loaded.reason_order, c.LEGACY_REASONS)
            self.assertIsNone(loaded.project_policy)
            self.assertEqual(demo.demo_only_reasons, ["heavy_dirt"])
            self.assertEqual(demo.thresholds["heavy_dirt"], 0.5)
            scores = [[0, 0.9, 0]] * 3
            self.assertEqual(c.decide(scores, demo.thresholds, demo.enabled, demo.reason_order)["reasons"],
                             ["heavy_dirt"])
            self.assertEqual(c.decide(scores, loaded.thresholds, loaded.enabled, loaded.reason_order)["status"], "OK")
            self.assertNotIn("heavy_dirt", loaded.enabled)
            crops = {camera: Image.new("RGB", (20, 40)) for camera in c.CAMERAS}
            result = loaded.predict(crops)
            self.assertEqual(result["reasons"], ["major_damage"])
            self.assertNotIn("mechanical_damage", result["reasons"])
            self.assertIsNone(result["project_policy"])
            self.assertTrue(result["legacy_reason_schema"])

    def test_checkpoint_schema_order_policy_and_output_shape_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            reason_path, _, _ = inference_checkpoints(Path(directory), legacy=False)
            original = torch.load(reason_path, weights_only=True)
            invalid = [{**original, "reason_schema_version": 3},
                       {**original, "reason_schema_version": True},
                       {k: v for k, v in original.items() if k != "reason_schema_version"},
                       {**original, "reason_order": list(reversed(c.REASONS))},
                       {**original, "project_policy": "primagaz_and_benegas"}]
            for state in invalid:
                torch.save(state, reason_path)
                with self.assertRaisesRegex(ValueError, "schema v2"):
                    c.BottleClassifier(reason_path)
            changed = copy.deepcopy(original)
            changed["model_state"]["head.weight"] = torch.zeros(3, 576)
            torch.save(changed, reason_path)
            with self.assertRaises(RuntimeError):
                c.BottleClassifier(reason_path)

    def test_new_training_rejects_legacy_dataset_before_features_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, triplet("old", "train"), legacy=True)
            with patch.object(training, "extract_features") as features:
                with self.assertRaisesRegex(ValueError, "human re-review"):
                    training.main(["train", "--data", str(root), "--output", str(root / "run")])
            features.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_v2_dataset_accepts_per_image_labels_and_unknown_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = triplet("photo", "train", (1, None, None, 0), (1, 0, 0, 1))[:1]
            rows[0].update(source="instructor_photo", camera_id="", primagaz_status=None)
            unknown = dict(rows[0], group_id="context", crop_id="context", training_approved=False,
                           reason_targets=[None] * 4, reason_loss_mask=[0] * 4)
            quarantined = dict(unknown, group_id="flagged", crop_id="flagged", quarantined=True)
            write_dataset(root, rows + [unknown, quarantined])
            _, loaded = c.load_dataset(root)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[2]["split"], "train")
            rows[0]["training_approved"] = False
            write_dataset(root, rows)
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                c.load_dataset(root)

    def test_primary_video_selection_excludes_photos_incomplete_and_quarantine(self):
        rows = triplet("video", "validation", (0,) * 4, (1,) * 4)
        rows[1]["primary_view"] = True
        photos = triplet("photos", "validation", (1,) * 4, (1,) * 4)
        for row in photos:
            row["source"] = "instructor_photo"
        incomplete = triplet("incomplete", "validation", (0,) * 4, (1,) * 4)[:2]
        flagged = triplet("flagged", "validation", (0,) * 4, (1,) * 4)
        flagged[0]["quarantined"] = True
        duplicate = triplet("duplicate", "validation", (0,) * 4, (1,) * 4)
        duplicate[2]["camera_id"] = "front"
        nonprimary = triplet("nonprimary", "validation", (0,) * 4, (1,) * 4)
        nonprimary[1]["primary_view"] = "0"
        excluded = triplet("excluded", "validation", (0,) * 4, (1,) * 4)
        excluded[1].update(primary_view=True, calibration_excluded=True)
        selected = c.primary_video_triplets(rows + photos + incomplete + flagged + duplicate + nonprimary + excluded)
        self.assertEqual(selected, rows)
        self.assertEqual(c.primary_video_triplets(photos), [])

    def test_known_v2_labels_reject_invalid_source_or_nonprimary_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source, camera in [("unverified", "front"), ("video", "top"), ("instructor_photo", "front")]:
                rows = triplet("bad", "train", (1, 0, 0, 0), (1,) * 4)[:1]
                rows[0].update(source=source, camera_id=camera)
                write_dataset(root, rows)
                with self.assertRaisesRegex(ValueError, "primary video camera or instructor photo"):
                    c.load_dataset(root)

    def test_photo_only_validation_cannot_calibrate_a_video_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = triplet("train", "train", (1, 0, 0, 1), (1,) * 4)
            photos = triplet("val", "validation", (0,) * 4, (1,) * 4)
            for row in photos:
                row["source"] = "instructor_photo"
                row["camera_id"] = ""
            write_dataset(root, rows + photos)
            with patch.object(training, "extract_features") as features:
                with self.assertRaisesRegex(ValueError, "primary-video VALIDATION"):
                    training.main(["train", "--data", str(root), "--output", str(root / "run")])
            features.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_four_reason_truth_preserves_unknown_and_missing_plant_reference(self):
        rows = triplet("video", "validation", (0, None, 0, 0), (1, 0, 1, 1))
        for row in rows:
            row.pop("primagaz_status")
        rows[1]["reason_targets"][0] = None
        rows[1]["reason_loss_mask"][0] = 0
        rows[2]["reason_targets"][3] = 1
        bottles = c.bottle_scores(rows, torch.zeros(3, 4))
        self.assertEqual(bottles[0]["truth"], [None, None, 0, 1])
        self.assertIsNone(bottles[0]["plant_status"])
        report = training.evaluate_bottles(bottles, {"wrong_brand_or_type": 0.5}, ["wrong_brand_or_type"])
        self.assertEqual(report["plant_agreement"]["unknown"], 1)
        self.assertEqual(report["plant_agreement"]["tn"], 0)
        self.assertEqual(report["plant_agreement"]["negatives"], 0)

    def test_binary_max_and_mean_use_all_three_views_and_one_feature_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            reason_path, binary_path, binary = inference_checkpoints(Path(directory))
            crops = {camera: Image.new("RGB", (20, 40)) for camera in c.CAMERAS}
            for binary_format in ("instructor_visual_ok_nok_v1", "gass_binary_experiment_v1"):
                binary.update(format=binary_format, model="mobilenet")
                torch.save(binary, binary_path)
                for aggregation in ("max", "mean"):
                    loaded = c.BottleClassifier(reason_path, binary_checkpoint=binary_path,
                                                binary_aggregation=aggregation)
                    features = torch.zeros(3, loaded.model.head.in_features)
                    features[:, 0] = torch.logit(torch.tensor([0.1, 0.9, 0.1]))
                    with patch.object(loaded.model.backbone, "forward", return_value=features) as encoder:
                        result = loaded.predict(crops)
                    encoder.assert_called_once()
                    # A single high side score rejects under max, but is diluted under mean.
                    self.assertEqual(result["status"], "NOK" if aggregation == "max" else "OK")
                    self.assertAlmostEqual(result["binary_score"], 0.9 if aggregation == "max" else 1.1 / 3)
                    self.assertEqual(result["binary_aggregation"], aggregation)
                    self.assertEqual(result["reason_status"], "NOK")
                    self.assertEqual(result["reasons"], ["major_damage"])
                    self.assertEqual(result["reason_disagreement"], aggregation == "mean")
                    self.assertFalse(result["reason_unresolved"])
                    self.assertEqual(result["decision_source"], f"binary_{aggregation}")
                    self.assertEqual(result["binary_threshold"], 0.5)
                    self.assertEqual(result["evidence_views"]["major_damage"], list(c.CAMERAS))
                    self.assertEqual(result["binary_scores"], dict(zip(c.CAMERAS, features[:, 0].sigmoid().tolist())))
                    with torch.no_grad():
                        loaded.model.head.bias.fill_(-4)
                    features[0, 0] = 4
                    with patch.object(loaded.model.backbone, "forward", return_value=features):
                        unresolved = loaded.predict(crops)
                        with torch.no_grad():
                            loaded.binary_head.bias.fill_(float("nan"))
                        with self.assertRaisesRegex(ValueError, "nonfinite scores"):
                            loaded.predict(crops)
                    self.assertEqual(unresolved["status"], "NOK")
                    self.assertEqual(unresolved["reasons"], [])
                    self.assertTrue(unresolved["reason_unresolved"])
                    self.assertTrue(unresolved["reason_disagreement"])
                    for missing in c.CAMERAS:
                        with self.assertRaisesRegex(ValueError, "all three"):
                            loaded.predict({name: image for name, image in crops.items() if name != missing})

    def test_binary_threshold_override_and_inclusive_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            reason_path, binary_path, _ = inference_checkpoints(Path(directory))
            crops = {camera: Image.new("RGB", (20, 40)) for camera in c.CAMERAS}
            for aggregation, cutoff, expected in [("max", 0.95, "OK"), ("mean", 0.3, "NOK"),
                                                   ("max", 0, "NOK"), ("mean", 1, "OK")]:
                loaded = c.BottleClassifier(reason_path, binary_checkpoint=binary_path,
                                            binary_aggregation=aggregation, nok_threshold=cutoff)
                features = torch.zeros(3, loaded.model.head.in_features)
                features[:, 0] = torch.logit(torch.tensor([0.1, 0.9, 0.1]))
                with patch.object(loaded.model.backbone, "forward", return_value=features):
                    result = loaded.predict(crops)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["binary_threshold"], cutoff)
            for aggregation in ("max", "mean"):
                loaded = c.BottleClassifier(reason_path, binary_checkpoint=binary_path,
                                            binary_aggregation=aggregation)
                with patch.object(loaded.model.backbone, "forward", return_value=torch.zeros_like(features)):
                    self.assertEqual(loaded.predict(crops)["status"], "NOK")  # Score equals cutoff.

    def test_binary_checkpoint_rejects_incompatible_or_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            reason_path, binary_path, binary = inference_checkpoints(Path(directory))
            invalid = [("format", "other"), ("image_size", 640), ("positive_label", "ok"),
                       ("threshold", -0.1), ("threshold", 1.1), ("threshold", float("nan")),
                       ("threshold", float("inf")), ("threshold", True)]
            for key, value in invalid:
                with self.subTest(key=key, value=value):
                    torch.save({**binary, key: value}, binary_path)
                    with self.assertRaises(ValueError):
                        c.BottleClassifier(reason_path, binary_checkpoint=binary_path)
            torch.save(binary, binary_path)
            for cutoff in (-0.1, 1.1, float("nan"), float("inf"), True):
                with self.assertRaises(ValueError):
                    c.BottleClassifier(reason_path, binary_checkpoint=binary_path, nok_threshold=cutoff)
            with self.assertRaisesRegex(ValueError, "max or mean"):
                c.BottleClassifier(reason_path, binary_checkpoint=binary_path, binary_aggregation="front")
            with self.assertRaisesRegex(ValueError, "require a binary checkpoint"):
                c.BottleClassifier(reason_path, nok_threshold=0.5)
            torch.save({**binary, "format": "gass_binary_experiment_v1", "model": "efficientnet"}, binary_path)
            with self.assertRaises(ValueError):
                c.BottleClassifier(reason_path, binary_checkpoint=binary_path)
            for key in ("backbone.features.0.0.weight", "backbone.features.0.1.running_mean",
                        "head.weight"):
                changed = copy.deepcopy(binary)
                changed["state_dict"][key] += 1 if key.startswith("backbone.") else float("nan")
                torch.save(changed, binary_path)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    c.BottleClassifier(reason_path, binary_checkpoint=binary_path)
            changed = copy.deepcopy(binary)
            changed["state_dict"]["head.weight"] = torch.zeros(2, 576)
            torch.save(changed, binary_path)
            with self.assertRaises(RuntimeError):
                c.BottleClassifier(reason_path, binary_checkpoint=binary_path)
            with self.assertRaisesRegex(ValueError, "demo-dirt"):
                c.BottleClassifier(reason_path, binary_checkpoint=binary_path, demo_dirt=True)

    def test_binary_cli_configuration_requires_model_and_finite_cutoff(self):
        images = ["--front", "a.jpg", "--back-left", "b.jpg", "--back-right", "c.jpg"]
        for args in [["--binary-aggregation", "max"], ["--nok-threshold", "0.5"],
                     ["--binary-checkpoint", "binary.pt", "--nok-threshold", "nan"],
                     ["--binary-checkpoint", "binary.pt", "--binary-aggregation", "front"]]:
            with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                c.main(["--checkpoint", "reason.pt", *images, *args])
            self.assertEqual(error.exception.code, 2)

    def test_preprocessing_preserves_both_ends_and_pads_sides(self):
        image = Image.new("RGB", (20, 100), "white")
        image.paste("red", (0, 0, 20, 10))
        image.paste("blue", (0, 90, 20, 100))
        tensor = c.preprocess(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        restored = tensor * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        restored += torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.assertTrue(torch.allclose(restored[:, 112, 0], torch.zeros(3), atol=1e-6))
        self.assertGreater(restored[0, 0, 112].item(), 0.99)
        self.assertGreater(restored[2, -1, 112].item(), 0.99)

    def test_unknown_and_disabled_labels_have_zero_gradient(self):
        rows = triplet("a", "train", (1, None, None), (1, 0, 0))
        targets, masks = c.targets_and_masks(rows, ["major_damage"], c.LEGACY_REASONS)
        logits = torch.zeros(3, 3, requires_grad=True)
        c.masked_loss(logits, targets, masks).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[:, 0] < 0).all())
        self.assertEqual(torch.count_nonzero(logits.grad[:, 1:]).item(), 0)
        with self.assertRaisesRegex(ValueError, "No known"):
            c.masked_loss(logits, targets, masks * 0)

    def test_any_view_can_supply_multiple_reasons_and_dirt_is_ignored(self):
        scores = [[0.1, 0.99, 0.1], [0.9, 0.99, 0.1], [0.1, 0.99, 0.8]]
        thresholds = dict(zip(c.LEGACY_REASONS, [0.5, None, 0.5]))
        result = c.decide(scores, thresholds, [c.LEGACY_REASONS[0], c.LEGACY_REASONS[2]], c.LEGACY_REASONS)
        self.assertEqual(result["status"], "NOK")
        self.assertEqual(result["evidence_views"], {"major_damage": ["back_left"],
                                                  "wrong_brand_or_model": ["back_right"]})
        result = c.decide([[0.1, 0.99, 0.1]] * 3, thresholds, [c.LEGACY_REASONS[0], c.LEGACY_REASONS[2]], c.LEGACY_REASONS)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["unchecked_reasons"], ["heavy_dirt"])
        with self.assertRaisesRegex(ValueError, "threshold"):
            c.decide(scores, {}, ["major_damage"], c.LEGACY_REASONS)
        with self.assertRaisesRegex(ValueError, "finite"):
            c.decide([[float("nan")] * 3] * 3, thresholds, ["major_damage"], c.LEGACY_REASONS)

    def test_bottle_truth_does_not_turn_hidden_views_into_negatives(self):
        rows = triplet("a", "validation")
        rows[0]["reason_targets"][0] = None
        rows[0]["reason_loss_mask"][0] = 0
        bottles = c.bottle_scores(rows, torch.zeros(3, 3), c.LEGACY_REASONS)
        self.assertIsNone(bottles[0]["truth"][0])
        rows[1]["reason_targets"][0] = 1
        self.assertEqual(c.bottle_scores(rows, torch.zeros(3, 3), c.LEGACY_REASONS)[0]["truth"][0], 1)

    def test_leaking_bottle_and_duplicate_camera_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = triplet("a", "train")
            rows[-1]["split"] = "test"
            write_dataset(root, rows, legacy=True)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                c.load_dataset(root, allow_legacy=True)
            rows = triplet("a", "train")
            rows[-1]["camera_id"] = "front"
            write_dataset(root, rows, legacy=True)
            with self.assertRaisesRegex(ValueError, "one crop per camera"):
                c.load_dataset(root, allow_legacy=True)

    def test_no_positive_test_bottles_means_undefined_recall(self):
        scores = [{"truth": [0, None, 0], "scores": [0.1, 0.99, 0.1], "plant_status": "NOK"}]
        report = training.evaluate_bottles(scores, {"major_damage": 0.5}, ["major_damage"], c.LEGACY_REASONS)
        self.assertIsNone(report["per_reason"]["major_damage"]["recall"])
        self.assertEqual(report["plant_agreement"]["fn"], 1)
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            training.calibrate(scores, ["major_damage"], c.LEGACY_REASONS)

    def test_backbone_and_batchnorm_stay_frozen(self):
        model = c.ReasonModel()
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.head.training)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_training_checkpoint_and_triplet_inference_without_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split in ("train", "validation"):
                for i, targets in enumerate(((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))):
                    rows.extend(triplet(f"{split}{i}", split, targets, (1, 0, 0, 1)))
            rows.extend(triplet("unopened", "test", (0, 0, 0, 0), (1, 0, 0, 1)))
            photo = dict(rows[0], group_id="photo", crop_id="photo", camera_id="",
                         source="instructor_photo", image_path="photo.jpg")
            context = dict(photo, group_id="unknown", crop_id="unknown", training_approved=False,
                           reason_targets=[None] * 4, reason_loss_mask=[0] * 4)
            flagged = dict(context, group_id="flagged", crop_id="flagged", quarantined=True)
            val_photo = dict(photo, group_id="val-photo", crop_id="val-photo", split="validation")
            rows.extend([photo, context, flagged, val_photo])
            write_dataset(root, rows)
            # Synthetic cached features test orchestration without external weights/images.
            def features(model, data, selected, device):
                self.assertTrue(all(r["split"] != "test" for r in selected))
                self.assertFalse(any(r["group_id"] in ("unknown", "flagged", "val-photo") for r in selected))
                if selected[0]["split"] == "train":
                    self.assertIn(photo, selected)
                return torch.randn(len(selected), model.head.in_features)

            with (patch.object(training, "ReasonModel", side_effect=lambda pretrained: c.ReasonModel()),
                  patch.object(training, "extract_features", side_effect=features)):
                report = training.main(["train", "--data", str(root), "--device", "cpu",
                                        "--epochs", "1", "--output", str(root / "run")])
            self.assertEqual(report["bottles"], 3)
            self.assertEqual(report["training_images"], 10)
            self.assertEqual(report["validation_excluded_images"], 1)
            self.assertFalse((root / "unopened_front.jpg").exists())
            loaded = c.BottleClassifier(root / "run" / "classifier.pt")
            self.assertEqual(loaded.reason_schema_version, 2)
            self.assertEqual(loaded.reason_order, c.REASONS)
            self.assertEqual(loaded.project_policy, c.PROJECT_POLICY)
            with self.assertRaisesRegex(ValueError, "legacy-only"):
                c.BottleClassifier(root / "run" / "classifier.pt", demo_dirt=True)
            self.assertIsNone(torch.load(root / "run" / "classifier.pt", weights_only=True)["thresholds"]["heavy_contamination"])
            with self.assertRaisesRegex(ValueError, "all three"):
                loaded.predict({"front": Image.new("RGB", (20, 40))})
            crops = {camera: Image.new("RGB", (20, 40), "white") for camera in c.CAMERAS}
            result = loaded.predict(crops, "example")
            self.assertEqual(result["bottle_id"], "example")
            self.assertIn(result["status"], ("OK", "NOK"))
            before = copy.deepcopy(loaded.model.state_dict())
            loaded.predict(crops)
            self.assertTrue(all(torch.equal(value, loaded.model.state_dict()[key])
                                for key, value in before.items()))


if __name__ == "__main__":
    unittest.main()
