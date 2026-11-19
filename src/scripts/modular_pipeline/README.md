# Modular three-camera pipeline

This folder contains a separate, self-contained version of the existing three-camera GASS pipeline. It does not import modules from `src/scripts/`; third-party dependencies and model files are the only external requirements.

## Files

- `m1_video_parser.py`: YOLO/ByteTrack tracking, crossing detection, global bottle matching, RGB crop extraction and `output.mp4` rendering.
- `m2_classifier.py`: visual classifier and optional binary classifier checkpoint loading and three-view inference.
- `m3_ocr.py`: EasyOCR, optional field/PARSeq detection and tare/approval field selection.
- `orchestrator.py`: runs the three modules, combines their evidence, makes the final bottle decision and exports reports.
- `pipeline_config.py`: camera, tracker, crop, rendering, OCR, classifier and reason configuration.
- `ocr_selection.json`: default OCR field-selection thresholds.

## Run

Direct script execution:

```powershell
python src/scripts/modular_pipeline/orchestrator.py --weights path/to/detector.pt --videos back_left.mp4 back_right.mp4 front.mp4 --output src/runs/modular_pipeline --no-display
```

Package-style execution:

```powershell
python -m src.scripts.modular_pipeline.orchestrator --weights path/to/detector.pt --videos back_left.mp4 back_right.mp4 front.mp4 --output src/runs/modular_pipeline --no-display
```

Optional module flags match the existing pipeline:

```text
--classifier <checkpoint>
--binary-classifier <checkpoint>
--ocr
--ocr-fields <field-detector.pt>
--ocr-selection-config <config.json>
--tracks-dir <track-csv-folder>
--retrack
--start-seconds <seconds>
--seconds <seconds>
--device <auto|cpu|mps|cuda:0>
--no-display
```

## Final decision

The orchestrator combines evidence with this minimal policy:

- Visual `NOK` or an expired approval year produces final `NOK`.
- Missing or uncertain OCR or visual evidence produces `INCOMPLETE`.
- A visual `OK`, accepted tare and valid approval year produce final `OK`.

## Outputs

- `output.mp4`: synchronized 2×2 annotated video.
- `predictions.json`: detailed bottle, module, crop and provenance records.
- `bottles.csv`: one summary row per bottle.

The CSV columns are `bottle_id`, `status`, `visual_status`, `visual_reasons`, `tare_kg`, `tare_status`, `approval_until_year`, `approval_status`, `approval_valid`, `ready_frame`, and `error`.

No `__init__.py` or `__main__.py` is required. Direct execution adds this folder temporarily to `sys.path`; package-style execution uses relative imports.
