import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.scripts import prepare_reason_dataset as p


class PrepareReasonDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.crosswalk = self.root / "crosswalk.csv"
        self.export = self.root / "export.json"
        self.output = self.root / "prepared-v2"
        self.records, self.tasks = [], []
        for project, split in {101: "train", 102: "validation", 103: "test"}.items():
            image = self.root / f"{project}.jpg"
            image.write_bytes(f"synthetic-image-{project}".encode())
            record = {"task_id": str(project), "project_id": str(project), "split": split,
                      "group_id": f"group-{project}", "review_id": f"crop-{project}",
                      "neutral_id": f"R{project}", "neutral_group": f"G{project}",
                      "image": f"gs://private/{project}.jpg", "image_path": str(image),
                      "sha256": p.sha256(image), "source": "video", "camera_id": "back_left",
                      "primary_view": "1", "label": "nok", "recording_id": "synthetic"}
            self.records.append(record)
            self.tasks.append({"id": project, "project": project, "meta": {}, "data": {
                "review_id": record["neutral_id"], "review_group": record["neutral_group"],
                "image": record["image"], "split": split, "camera": "back_left"},
                "annotations": []})

    def write_inputs(self):
        with self.crosswalk.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        self.export.write_text(json.dumps(self.tasks))

    def prepare(self):
        self.write_inputs()
        metadata = p.prepare([self.export], self.crosswalk, self.output)
        rows = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]
        return metadata, rows

    def review(self, answers=("present", "unknown", "absent", "unknown")):
        return {"id": 55, "completed_by": 1, "was_cancelled": False,
                "result": [{"from_name": reason, "to_name": "image", "type": "choices",
                            "value": {"choices": [answer]}}
                           for reason, answer in zip(p.REASONS, answers)]}

    def assert_rejected(self, message):
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, message):
            p.prepare([self.export], self.crosswalk, self.output)
        self.assertFalse(self.output.exists())

    def test_unknown_context_and_source_nok_do_not_create_labels(self):
        metadata, rows = self.prepare()
        self.assertEqual(metadata["reason_schema_version"], 2)
        self.assertEqual(metadata["enabled_reason_outputs"], list(p.REASONS))
        self.assertEqual(metadata["known_label_images"], 0)
        for row in rows:
            self.assertEqual(row["reason_targets"], [None] * 4)
            self.assertEqual(row["reason_loss_mask"], [0] * 4)
            self.assertFalse(row["training_approved"])
            self.assertIsNone(row["primagaz_status"])
            self.assertEqual(row["original_source_label"], "nok")

    def test_enabled_reasons_requires_at_least_one_distinct_canonical_output(self):
        self.write_inputs()
        for enabled in ([], ["not_a_reason"], [p.REASONS[0]] * 2):
            with self.subTest(enabled=enabled), self.assertRaisesRegex(ValueError, "at least one distinct"):
                p.prepare([self.export], self.crosswalk, self.output, enabled)
            self.assertFalse(self.output.exists())
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            p.main(["--exports", str(self.export), "--crosswalk", str(self.crosswalk),
                    "--output", str(self.output), "--enabled-reasons"])
        self.assertEqual(error.exception.code, 2)
        metadata = p.prepare([self.export], self.crosswalk, self.output, [p.REASONS[0]])
        self.assertEqual(metadata["enabled_reason_outputs"], [p.REASONS[0]])

    def test_explicit_answers_and_owner_origin_preserved(self):
        self.tasks[0]["annotations"] = [self.review()]
        self.tasks[0]["meta"] = {"owner_chat_review": {"inspection_id": "synthetic-review"},
                                  "project_visual_status": "NOK"}
        self.records[0]["label"] = "ok"
        _, rows = self.prepare()
        row = rows[0]
        self.assertEqual(row["reason_targets"], [1, None, 0, None])
        self.assertEqual(row["reason_loss_mask"], [1, 0, 1, 0])
        self.assertTrue(row["training_approved"])
        self.assertEqual(row["original_source_label"], "ok")
        self.assertEqual(row["project_visual_status"], "NOK")
        self.assertEqual(row["task_meta"]["owner_chat_review"]["inspection_id"], "synthetic-review")
        self.assertEqual(row["label_studio"]["annotation"]["id"], 55)

    def test_quarantine_keeps_split_and_raw_review_but_masks_everything(self):
        self.tasks[0]["annotations"] = [self.review()]
        self.tasks[0]["meta"] = {"exclude_from_fitting": True}
        _, rows = self.prepare()
        self.assertTrue(rows[0]["quarantined"])
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[0]["reason_loss_mask"], [0] * 4)
        self.assertEqual(rows[0]["reviewed_reason_loss_mask"], [1, 0, 1, 0])
        self.assertFalse(rows[0]["training_approved"])

    def test_cancelled_annotation_not_truth(self):
        cancelled = self.review()
        cancelled["was_cancelled"] = True
        self.tasks[0]["annotations"] = [cancelled]
        _, rows = self.prepare()
        self.assertFalse(rows[0]["annotation_reviewed"])
        self.assertEqual(rows[0]["label_studio"]["cancelled_annotation_count"], 1)

    def test_multiple_annotations_require_adjudication(self):
        self.tasks[0]["annotations"] = [self.review(), self.review()]
        self.assert_rejected("adjudication")

    def test_missing_unknown_answer_is_not_silently_masked(self):
        review = self.review()
        review["result"].pop()
        self.tasks[0]["annotations"] = [review]
        self.assert_rejected("all four")

    def test_legacy_reason_annotation_rejected(self):
        review = self.review()
        review["result"][0]["from_name"] = "major_damage"
        self.tasks[0]["annotations"] = [review]
        self.assert_rejected("unexpected")

    def test_explicit_old_schema_or_different_policy_rejected(self):
        self.tasks[0]["meta"] = {"reason_schema_version": 1}
        self.assert_rejected("schema or project policy")
        self.tasks[0]["meta"] = {"project_policy": "different-policy"}
        self.assert_rejected("schema or project policy")

    def test_missing_project_export_rejected(self):
        self.tasks.pop()
        self.assert_rejected("Full exports")

    def test_identity_and_hash_mismatch_rejected(self):
        self.tasks[0]["data"]["image"] = "gs://wrong/image.jpg"
        self.assert_rejected("identity")
        self.tasks[0]["data"]["image"] = self.records[0]["image"]
        Path(self.records[0]["image_path"]).write_bytes(b"changed")
        self.assert_rejected("SHA-256")

    def test_group_and_image_split_leakage_rejected(self):
        self.records[1]["group_id"] = self.records[0]["group_id"]
        self.assert_rejected("multiple splits")
        self.records[1]["group_id"] = "new-group"
        self.records[1]["sha256"] = self.records[0]["sha256"]
        self.records[1]["image_path"] = self.records[0]["image_path"]
        self.assert_rejected("same image")

    def test_historical_representative_flag_does_not_drop_primary_side_cameras(self):
        for record, task, camera, flag in zip(self.records, self.tasks,
                                               ("front", "back_left", "back_right"), ("1", "0", "0")):
            record.update(camera_id=camera, primary_view=flag)
            task["data"]["camera"] = camera
        _, rows = self.prepare()
        self.assertTrue(all(row["primary_view"] for row in rows))
        self.assertEqual([row["source_primary_view"] for row in rows], ["1", "0", "0"])

    def test_auxiliary_membership_and_separate_calibration_exclusion(self):
        self.records[0]["camera_id"] = "top"
        self.tasks[0]["data"]["camera"] = "top"
        self.tasks[1]["meta"] = {"calibration_excluded": True}
        _, rows = self.prepare()
        self.assertFalse(rows[0]["primary_view"])
        self.assertTrue(rows[1]["primary_view"])
        self.assertTrue(rows[1]["calibration_excluded"])

    def test_known_supervision_on_auxiliary_or_unknown_video_camera_rejected(self):
        self.tasks[0]["annotations"] = [self.review()]
        for camera in ("top", "unknown"):
            self.records[0]["camera_id"] = camera
            self.tasks[0]["data"]["camera"] = camera
            self.assert_rejected("known supervision")

    def test_known_photo_supervision_cannot_claim_a_video_camera(self):
        self.records[0]["source"] = "instructor_photo"
        self.tasks[0]["annotations"] = [self.review()]
        self.assert_rejected("known supervision")

    def test_native_standalone_photo_display_camera_is_not_a_training_camera(self):
        self.records[0].update(source="instructor_photo", camera_id="", primary_view="0")
        self.tasks[0]["data"]["camera"] = "standalone photo"
        self.tasks[0]["annotations"] = [self.review()]
        _, rows = self.prepare()
        self.assertEqual(rows[0]["camera_id"], "")
        self.assertFalse(rows[0]["primary_view"])
        self.assertTrue(rows[0]["training_approved"])

    def test_output_immutable_and_images_unchanged(self):
        originals = [Path(r["image_path"]).read_bytes() for r in self.records]
        self.prepare()
        with self.assertRaises(FileExistsError):
            p.prepare([self.export], self.crosswalk, self.output)
        self.assertEqual(originals, [Path(r["image_path"]).read_bytes() for r in self.records])


if __name__ == "__main__":
    unittest.main()
