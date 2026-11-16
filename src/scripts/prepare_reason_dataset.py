"""Prepare schema-v2 reason labels from complete native Label Studio JSON exports.

The private crosswalk supplies identity and frozen splits, never reason truth.
Original images remain in place; their hashes are checked before output is saved.
The crosswalk's historical primary_view selects a representative image. V2
primary_view means membership of the three video cameras; the raw flag is kept
as source_primary_view. Task meta calibration_excluded can exclude a view.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

if __package__:
    from .reason_schema import PROJECT_POLICY, REASONS, REASON_SCHEMA_VERSION
else:
    from reason_schema import PROJECT_POLICY, REASONS, REASON_SCHEMA_VERSION

CAMERAS = {"front", "back_left", "back_right"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_name(value):
    return "test" if value == "final_test" else value


def annotation_values(task):
    """A completed review must explicitly answer each canonical reason once."""
    annotations = task.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"Task {task['id']}: annotations must be a list.")
    active = [a for a in annotations if a.get("was_cancelled") is not True]
    if len(active) > 1:
        raise ValueError(f"Task {task['id']}: multiple completed annotations need adjudication.")
    if not active:
        return [None] * len(REASONS), [0] * len(REASONS), None
    annotation = active[0]
    if not annotation.get("completed_by"):
        raise ValueError(f"Task {task['id']}: missing reviewer identity.")
    answers = {}
    for result in annotation.get("result", []):
        name = result.get("from_name")
        if name == "notes" and result.get("type") == "textarea":
            continue
        if (name not in REASONS or name in answers or result.get("type") != "choices"
                or result.get("to_name") != "image"):
            raise ValueError(f"Task {task['id']}: unexpected or duplicate reason control {name!r}.")
        choices = result.get("value", {}).get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or choices[0] not in (
                "present", "absent", "unknown"):
            raise ValueError(f"Task {task['id']}: invalid answer for {name}.")
        answers[name] = choices[0]
    if set(answers) != set(REASONS):
        raise ValueError(f"Task {task['id']}: all four reason answers are required.")
    targets = [{"present": 1, "absent": 0, "unknown": None}[answers[r]] for r in REASONS]
    return targets, [int(value is not None) for value in targets], annotation


def prepare(exports, crosswalk, output, enabled_reasons=REASONS):
    output, crosswalk = Path(output), Path(crosswalk)
    if output.exists():
        raise FileExistsError(f"Choose a fresh output folder: {output}")
    if (not enabled_reasons or len(set(enabled_reasons)) != len(enabled_reasons)
            or not set(enabled_reasons) <= set(REASONS)):
        raise ValueError("Choose at least one distinct canonical reason.")
    with crosswalk.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    required = {"task_id", "project_id", "split", "group_id", "review_id", "neutral_id",
                "neutral_group", "image", "image_path", "sha256", "source", "camera_id"}
    if not records or not required <= records[0].keys():
        raise ValueError("Crosswalk is empty or missing identity/split/source columns.")
    expected, projects = {}, {}
    for record in records:
        task_id, project = int(record["task_id"]), int(record["project_id"])
        split = split_name(record["split"])
        if split not in ("train", "validation", "test") or projects.setdefault(project, split) != split:
            raise ValueError(f"Crosswalk task {task_id}: each project must have one frozen split.")
        if task_id in expected:
            raise ValueError(f"Duplicate crosswalk task {task_id}.")
        expected[task_id] = record
    if len(projects) != 3 or set(projects.values()) != {"train", "validation", "test"}:
        raise ValueError("Crosswalk must contain exactly one project for each train/validation/test split.")
    tasks = {}
    for path in map(Path, exports):
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            raise ValueError(f"Expected a native JSON task list: {path}")
        for task in content:
            task_id = task["id"]
            if type(task_id) is not int or task_id in tasks or task_id not in expected:
                raise ValueError(f"Unexpected or duplicate exported task {task_id}.")
            tasks[task_id] = task
    if tasks.keys() != expected.keys():
        raise ValueError(f"Full exports required; missing {len(expected.keys() - tasks.keys())} tasks.")
    rows, groups, hashes, crop_ids = [], {}, {}, set()
    for task_id, record in expected.items():
        task = tasks[task_id]
        project, split = int(record["project_id"]), split_name(record["split"])
        data, meta = task.get("data", {}), task.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"Task {task_id}: meta must be an object.")
        if (meta.get("reason_schema_version", REASON_SCHEMA_VERSION) != REASON_SCHEMA_VERSION
                or meta.get("project_policy", PROJECT_POLICY) != PROJECT_POLICY):
            raise ValueError(f"Task {task_id}: reason schema or project policy differs from v2.")
        checks = {"review_id": record["neutral_id"], "review_group": record["neutral_group"],
                  "image": record["image"], "camera": record["camera_id"] or "standalone photo"}
        if task.get("project") != project or any(data.get(k) != v for k, v in checks.items()):
            raise ValueError(f"Task {task_id}: export/crosswalk identity or project mismatch.")
        if split_name(data.get("split")) != split:
            raise ValueError(f"Task {task_id}: exported split differs from frozen crosswalk.")
        group, image_hash = record["group_id"], record["sha256"]
        if not group or groups.setdefault(group, split) != split:
            raise ValueError(f"Group appears in multiple splits: {group}")
        if hashes.setdefault(image_hash, split) != split:
            raise ValueError("The same image appears in multiple splits.")
        image_path = Path(record["image_path"]).expanduser().resolve()
        if sha256(image_path) != image_hash:
            raise ValueError(f"Task {task_id}: local image SHA-256 differs from frozen crosswalk.")
        if record["source"] not in ("video", "instructor_photo"):
            raise ValueError(f"Task {task_id}: unsupported source.")
        targets, masks, annotation = annotation_values(task)
        quarantined = (meta.get("exclude_from_fitting") is True
                       or meta.get("data_quality_status") == "quarantined")
        raw_targets, raw_masks = targets[:], masks[:]
        if quarantined:
            targets, masks = [None] * len(REASONS), [0] * len(REASONS)
        if any(masks) and ((record["source"] == "video" and record["camera_id"] not in CAMERAS)
                           or (record["source"] == "instructor_photo" and record["camera_id"])):
            raise ValueError(f"Task {task_id}: known supervision needs a primary video camera "
                             "or a standalone photo with blank camera_id.")
        crop_id = record.get("crop_id") or record["review_id"]
        if crop_id in crop_ids:
            raise ValueError(f"Duplicate crop identity: {crop_id}")
        crop_ids.add(crop_id)
        rows.append({
            "crop_id": crop_id, "group_id": group, "split": split,
            "image_path": str(image_path), "sha256": image_hash,
            "source": record["source"], "camera_id": record["camera_id"],
            "primary_view": record["source"] == "video" and record["camera_id"] in CAMERAS,
            "calibration_excluded": meta.get("calibration_excluded") is True,
            "source_primary_view": record.get("primary_view"),
            "recording_id": record.get("recording_id"), "primagaz_status": None,
            "reason_targets": targets, "reason_loss_mask": masks,
            "training_approved": any(masks),
            "annotation_reviewed": annotation is not None, "quarantined": quarantined,
            "reviewed_reason_targets": raw_targets, "reviewed_reason_loss_mask": raw_masks,
            "project_visual_status": meta.get("project_visual_status"),
            "original_source_label": record.get("label"),
            "source_provenance": dict(record), "task_meta": meta,
            "label_studio": {"project_id": project, "task_id": task_id,
                             "annotation": annotation,
                             "cancelled_annotation_count": sum(a.get("was_cancelled") is True
                                                               for a in task.get("annotations", []))},
        })
    metadata = {"dataset_version": output.name, "reason_schema_version": REASON_SCHEMA_VERSION,
                "reason_order": list(REASONS), "project_policy": PROJECT_POLICY,
                "manifest": "manifest.jsonl", "enabled_reason_outputs": list(enabled_reasons),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "split_counts": dict(Counter(r["split"] for r in rows)),
                "reviewed_images": sum(r["annotation_reviewed"] for r in rows),
                "known_label_images": sum(any(r["reason_loss_mask"]) for r in rows),
                "quarantined_images": sum(r["quarantined"] for r in rows),
                "image_storage": "Verified absolute original paths; source files are unchanged.",
                "provenance": {"crosswalk": str(crosswalk.resolve()), "crosswalk_sha256": sha256(crosswalk),
                               "exports": [{"path": str(Path(p).resolve()), "sha256": sha256(p)}
                                           for p in exports]},
                "note": "Exported labels do not establish training or calibration readiness. "
                        "No reasons are inferred from source class; plant outcomes are unset."}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (stage / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (stage / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, nargs="+", required=True,
                        help="Complete native JSON exports with download_all_tasks=true for all three split projects.")
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New versioned folder; never overwritten.")
    parser.add_argument("--enabled-reasons", nargs="+", choices=REASONS, default=list(REASONS),
                        help="Requested trainer outputs; default all four. Training checks coverage separately.")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.exports, args.crosswalk, args.output, args.enabled_reasons)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Preparation failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
