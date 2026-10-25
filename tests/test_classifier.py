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
             "training_approved": True, "primagaz_status": "NOK" if 1 in targets else "OK",
             "reason_targets": list(targets), "reason_loss_mask": list(masks)}
            for camera in c.CAMERAS]


def write_dataset(root, rows):
    metadata = {"manifest": "manifest.jsonl", "reason_order": list(c.REASONS),
                "enabled_reason_outputs": [c.REASONS[0], c.REASONS[2]], "dataset_version": "synthetic"}
    (root / "dataset.json").write_text(json.dumps(metadata))
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


class ClassifierTests(unittest.TestCase):
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
        targets, masks = c.targets_and_masks(rows, ["major_damage"])
        logits = torch.zeros(3, 3, requires_grad=True)
        c.masked_loss(logits, targets, masks).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[:, 0] < 0).all())
        self.assertEqual(torch.count_nonzero(logits.grad[:, 1:]).item(), 0)
        with self.assertRaisesRegex(ValueError, "No known"):
            c.masked_loss(logits, targets, masks * 0)

    def test_any_view_can_supply_multiple_reasons_and_dirt_is_ignored(self):
        scores = [[0.1, 0.99, 0.1], [0.9, 0.99, 0.1], [0.1, 0.99, 0.8]]
        thresholds = dict(zip(c.REASONS, [0.5, None, 0.5]))
        result = c.decide(scores, thresholds, [c.REASONS[0], c.REASONS[2]])
        self.assertEqual(result["status"], "NOK")
        self.assertEqual(result["evidence_views"], {"major_damage": ["back_left"],
                                                  "wrong_brand_or_model": ["back_right"]})
        result = c.decide([[0.1, 0.99, 0.1]] * 3, thresholds, [c.REASONS[0], c.REASONS[2]])
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["unchecked_reasons"], ["heavy_dirt"])
        with self.assertRaisesRegex(ValueError, "threshold"):
            c.decide(scores, {}, ["major_damage"])
        with self.assertRaisesRegex(ValueError, "finite"):
            c.decide([[float("nan")] * 3] * 3, thresholds, ["major_damage"])

    def test_bottle_truth_does_not_turn_hidden_views_into_negatives(self):
        rows = triplet("a", "validation")
        rows[0]["reason_targets"][0] = None
        rows[0]["reason_loss_mask"][0] = 0
        bottles = c.bottle_scores(rows, torch.zeros(3, 3))
        self.assertIsNone(bottles[0]["truth"][0])
        rows[1]["reason_targets"][0] = 1
        self.assertEqual(c.bottle_scores(rows, torch.zeros(3, 3))[0]["truth"][0], 1)

    def test_leaking_bottle_and_duplicate_camera_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = triplet("a", "train")
            rows[-1]["split"] = "test"
            write_dataset(root, rows)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                c.load_dataset(root)
            rows = triplet("a", "train")
            rows[-1]["camera_id"] = "front"
            write_dataset(root, rows)
            with self.assertRaisesRegex(ValueError, "one crop per camera"):
                c.load_dataset(root)

    def test_no_positive_test_bottles_means_undefined_recall(self):
        scores = [{"truth": [0, None, 0], "scores": [0.1, 0.99, 0.1], "plant_status": "NOK"}]
        report = training.evaluate_bottles(scores, {"major_damage": 0.5}, ["major_damage"])
        self.assertIsNone(report["per_reason"]["major_damage"]["recall"])
        self.assertEqual(report["plant_agreement"]["fn"], 1)
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            training.calibrate(scores, ["major_damage"])

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
                for i, targets in enumerate(((0, 0, 0), (1, 0, 0), (0, 0, 1))):
                    rows.extend(triplet(f"{split}{i}", split, targets))
            rows.extend(triplet("unopened", "test"))
            write_dataset(root, rows)
            # Synthetic cached features test orchestration without external weights/images.
            def features(model, data, selected, device):
                self.assertTrue(all(r["split"] != "test" for r in selected))
                return torch.randn(len(selected), model.head.in_features)

            with (patch.object(training, "ReasonModel", side_effect=lambda pretrained: c.ReasonModel()),
                  patch.object(training, "extract_features", side_effect=features)):
                report = training.main(["train", "--data", str(root), "--device", "cpu",
                                        "--epochs", "1", "--output", str(root / "run")])
            self.assertEqual(report["bottles"], 3)
            self.assertFalse((root / "unopened_front.jpg").exists())
            loaded = c.BottleClassifier(root / "run" / "classifier.pt")
            demo = c.BottleClassifier(root / "run" / "classifier.pt", demo_dirt=True)
            self.assertNotIn("heavy_dirt", loaded.enabled)
            self.assertIn("heavy_dirt", demo.enabled)
            self.assertEqual(demo.thresholds["heavy_dirt"], 0.5)
            self.assertEqual(demo.demo_only_reasons, ["heavy_dirt"])
            self.assertEqual(loaded.demo_only_reasons, [])
            dirt_scores = [[0, 0.9, 0]] * 3
            self.assertEqual(c.decide(dirt_scores, demo.thresholds, demo.enabled)["reasons"], ["heavy_dirt"])
            self.assertEqual(c.decide(dirt_scores, loaded.thresholds, loaded.enabled)["status"], "OK")
            self.assertIsNone(torch.load(root / "run" / "classifier.pt", weights_only=True)["thresholds"]["heavy_dirt"])
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
