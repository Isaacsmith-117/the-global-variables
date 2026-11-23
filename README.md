# GASS: gas-cylinder detection and tracking

**The Global Variables:** Isaac Smith, Sebastian Albu, Cosmin Brinda,
Emilija Rimselite, and Wood de Bock.

Fine-tune YOLOv8 nano on CylinDeRS, evaluate the detector, and track cylinders
in local video using ByteTrack. A separate WS5 classifier checks matched bottle
crops for a visual OK/NOK decision and supporting rejection reasons.

## Layout

```text
src/scripts/       Detector/classifier training, evaluation, tracking, OCR, and shared config
src/models/        Selected shared weights
src/runs/          Generated outputs (ignored)
tests/             Local verification checks
data/              Local datasets and footage (ignored)
requirements.txt   Python dependencies
AGENTS.md          Team working agreements
```

Local data currently uses `data/CylinDeRS.v1i.yolov11/`,
`data/primagaz_images/`, and `data/primagaz_vids/`. Plant photos need annotation
before they become training examples. Keep private Primagaz media local;
do not commit it or send it to external AI services.

## Models

- **[yolov8n.pt](src/models/yolov8n.pt):** upstream Ultralytics YOLOv8 nano base
  model, used to start training; it is not fine-tuned for gas cylinders.
- **[2026-10-10_isaac_cylinders-yolov8n-50e.pt](src/models/2026-10-10_isaac_cylinders-yolov8n-50e.pt):**
  Isaac's evaluated gas-cylinder baseline, saved October 10, 2026. Trained on
  CylinDeRS v1 for 50 epochs at image size 416, batch size 8, seed 0, using MPS
  and AdamW. This is the best checkpoint (epoch 49), not the final checkpoint.

| Split | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Validation (2,246 images) | 88.0% | 78.0% | 86.4% | 61.4% |
| Test (100 images) | 88.3% | 76.7% | 87.3% | 67.7% |

- **[2026-10-24_cosmin_cylinders-yolo26s-70e.pt](src/models/2026-10-24_cosmin_cylinders-yolo26s-70e.pt):**
  Cosmin's YOLO26 small gas-cylinder detector, saved October 24, 2026. Trained
  from `yolo26s.pt` on CylinDeRS v1 with the Colab notebook for 70 epochs at
  image size 480, batch size 32, seed 0, using AdamW. This is the best
  checkpoint (epoch 65), not the final checkpoint. Scores below use image size
  480; the scripts default to 416.

| Split | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Validation (during training) | 92.3% | 86.1% | 92.8% | 69.7% |
| Test (100 images) | 81.6% | 86.8% | 90.5% | 68.7% |

These are CylinDeRS detector scores. Conveyor tracking accuracy, ID stability,
and counting reliability remain unverified. Only the selected models above are
shared; quick runs, fine-tuned candidates, and other checkpoints stay local.

## Setup

Use Python 3.13. From the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.13 -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1`. Select `.venv` as the IDE interpreter.
Scripts automatically select MPS, CUDA, or CPU; use `--device cpu` to override.

## Run

Run these commands from the project root with the environment activated.
Keep the dataset's training, validation, and test images separate. Pass
`--data` explicitly for the current dataset; the default is `data/data.yaml`.

```bash
python src/scripts/model.py --mode quick --data data/CylinDeRS.v1i.yolov11/data.yaml
python src/scripts/model.py --mode full --data data/CylinDeRS.v1i.yolov11/data.yaml
python src/scripts/evaluate.py --weights src/models/2026-10-10_isaac_cylinders-yolov8n-50e.pt --data data/CylinDeRS.v1i.yolov11/data.yaml
python src/scripts/track.py --weights src/models/2026-10-10_isaac_cylinders-yolov8n-50e.pt --source data/primagaz_vids/filling_plant_bolivia.webm
```

Quick training uses 3 epochs and 10% of training data; full training uses up to
50 epochs and all training data, with patience 10. Training loads the shared
base model and saves new weights under `src/runs/`. It does not replace shared
weights. Evaluation defaults to the held-out test split; use `--split val`
for validation.

Tracking processes every decoded frame and saves an annotated video under
`src/runs/tracking/preview/`, then `preview-2/`, etc. WebM input is supported.
Output video has no audio; its format is chosen by Ultralytics for the platform.
Use `--output` for a different output root. There is no CSV export or counting.
IDs can change after occlusion or re-entry, so they are not a unique bottle count.

### Colab training (YOLO26)

`src/scripts/train_colab_yolo26.ipynb` fine-tunes YOLO26 small (`yolo26s.pt`)
on a free Colab T4 GPU, with the same steps and settings as our first Colab
model. Upload it to Google Drive and open it with Colab, set `DATASET_ZIP` in
cell 4 to the dataset zip in Drive, then choose **Runtime → Run all**. Results
are saved to `My Drive/gas_cylinders_runs/yolo26s/`; the last cell explains how
to use `best.pt` with the scripts above. Set `MODEL = "yolo26n.pt"` for nano.

### Reviewable video crop dataset (issue #32)

Extract one recording at a time with explicit source paths and a new output
folder. Video names must start with the recording ID and identify `front`,
`back_left`, `back_right`, or `top`. The workbook name must start with the
recording ID followed by `_ground_truth`. Its first sheet supplies the references.

```bash
python src/scripts/extract_dataset.py --recording-id 14_32 --truth data/primagaz_vids/14_32/14_32_ground_truth_ai_app.xlsx --videos data/primagaz_vids/14_32/14_32_back_left_cropped.mp4 data/primagaz_vids/14_32/14_32_back_right_cropped.mp4 data/primagaz_vids/14_32/14_32_front_cropped.mp4 --output data/derived/14_32-review-v1
```

Add `--tracks-dir` to reuse existing CSVs named after the exact video stems.
This directory must contain every requested camera's CSV. Shape and frame checks
reject incompatible tracks. For older CSVs, independently confirm which video
produced them before reuse. Each run records the source and track hashes in
`run.json`. `complete` becomes true only after the manifest and review export
finish. A failed run keeps `complete` false; choose a fresh folder for its retry.
Use `--pilot 16` for a small review sample selected using candidate plant labels.

The output has crops, source frames, debug views, contact sheets, `manifest.csv`,
and `review.html`. Unverified crops go in `unknown` folders. Crossing order only
fills the `candidate_*` reference columns; `--id-offset` adjusts that proposal.
Review camera grouping and workbook correspondence before using plant labels.

To preserve reviewed matches across reruns, pass `--matches path/to/matches.csv`.
Each row confirms one camera crossing against an ID in this recording's workbook.
The CSV needs these columns:

```csv
recording_id,camera_id,track_id,frame_index,spreadsheet_id
```

Copy the track and frame from the review manifest and fill the workbook ID only
after verifying the physical bottle. The extractor rejects duplicate matches,
missing workbook IDs, and matches that do not identify a crossing in the supplied
tracks. Confirmed rows fill `bottle_id`, `source_row`, `spreadsheet_id`, and the
original `primagaz_status`, and set `match_verified=1`. Confirmed camera views are
grouped by workbook ID in the review. Rejection-reason labels remain `unknown`
and `review_status` remains `pending` for human annotation. Keep duplicates and
all views of the same bottle together when preparing training splits.

### Visual classifier and reason records (WS5, issues #17 and #45)

The visual decision is binary `OK`/`NOK`. Four independent reason fields support
the record: `corrosion`, `mechanical_damage`, `heavy_contamination`, and
`wrong_brand_or_type`. The project accepts **PrimaGaz only**, using the agreed
type examples; color alone does not establish brand or type. OCR supplies separate
text evidence. A visual OK does not certify every plant requirement.

Use reason schema **2** and project policy **`primagaz_only_v1`** for new labels
and training. Each image needs Present, Absent or Unknown for every reason.
Several reasons may be present together. Unknown labels are masked, never treated
as absent. Source visual labels, project decisions and `Primagaz_status` remain
separate facts.

`classifier.py` resizes each crop proportionally, pads it to 224×224 and extracts
frozen MobileNetV3-Small features. Small linear heads score binary NOK and the
reasons. Whole-bottle inference requires matched `front`, `back_left` and
`back_right` crops; reasons retain the views supporting them. Standalone photos
can supply reviewed reason training examples. Validation cutoff selection uses
complete three-camera bottle groups, matching the video decision.

Export the three reason projects separately and prepare a new private dataset:

```bash
python src/scripts/prepare_reason_dataset.py --exports train.json validation.json final-test.json --crosswalk private-crosswalk.csv --output data/prepared/visual-reasons-v2
python src/scripts/train_classifier.py train --data data/prepared/visual-reasons-v2 --output src/runs/classifier/reasons-v2
```

The exporter validates project/split, image hashes and complete answer fields.
Training consumes only approved TRAIN rows. Quarantined or unreviewed rows do not
become negative examples. Keep every view of a physical bottle in one split.
Validation loss, saved-weight selection and each enabled reason's cutoff use
eligible complete video validation triplets. Training stops if an enabled reason
lacks positive or negative support; disabled outputs remain unchecked.
Scores are model scores, not calibrated probabilities. Test evaluation is a
separate command, used after the model and cutoffs are fixed:

```bash
python src/scripts/train_classifier.py evaluate --data data/prepared/visual-reasons-v2 --checkpoint src/runs/classifier/reasons-v2/classifier.pt
python src/scripts/classifier.py --checkpoint src/runs/classifier/reasons-v2/classifier.pt --front path/to/front.jpg --back-left path/to/back_left.jpg --back-right path/to/back_right.jpg --bottle-id example
```

API callers pass RGB PIL images or paths to `BottleClassifier.predict()`; convert
OpenCV BGR crops to RGB first. Incomplete triplets are rejected. Evaluation keeps
per-reason counts, unknown coverage and plant agreement separate. Checkpoints and
reports stay in ignored `src/runs/`; support for schema 2 does not mean a trained
four-reason checkpoint has been selected or validated.

Historical schema-1 checkpoints retain their original `major_damage`,
`heavy_dirt` and `wrong_brand_or_model` meanings during inference. Their damage
positives included corrosion, so renaming that output to mechanical damage would
misstate what was learned. New training requires schema 2; preserve old labels,
weights and reports for reproducibility.

### Annotated three-camera classifier video

`tricam_offline.py --classifier` connects the classifier to the existing 2×2
camera display. Video order is **back-left, back-right, front**. Each bottle is
classified once using its three line-crossing crops. All crops come from the
videos and tracker boxes; plant labels are not used to generate predictions.
The display shows grey Inspecting boxes, green OK boxes, red NOK boxes with
reasons, and a panel counting completed inspections once per bottle. Unsupported
reason outputs remain marked unchecked. Results appear after all three crossing crops are
available in the synchronized playback.

```bash
python src/scripts/tricam_offline.py --classifier path/to/reasons-v2.pt --tracks-dir data/dataset/tracks --start-seconds 315 --seconds 45 --output src/runs/tricam_classifier/demo --device cpu --no-display
```

Without `--no-display`, a preview window is shown while rendering; press Q to stop.
The output folder receives `output.mp4` and `predictions.json` with decisions,
scores, evidence views, crop frames/boxes and clip counts. Existing output files
are never overwritten. Omit `--tracks-dir` to run the detector and create tracks;
use `--videos` to provide three other recordings. Omit `--seconds` for the full
synchronized recording. Saved tracks must correspond to the supplied videos.

For a historical schema-1 dirt integration demo, add `--demo-dirt`. With an old two-reason
checkpoint this includes its untrained dirt output at a fixed 0.5 cutoff;
it is not evidence of dirt-detection performance. The overlay uses the normal
`heavy dirt` reason text. Prediction JSON records `demo_only_reasons` and the run
records `demo_dirt_requested`. The checkpoint, dataset and normal default behaviour
are unchanged. The flag is rejected for schema 2: heavy contamination needs its
own reviewed labels and new weights.

#### Independent binary verdict and supporting reasons (issue #43)

Add `--binary-classifier path/to/binary.pt` alongside `--classifier path/to/reasons.pt`
to use the binary model on **all three views**. Choose `--binary-aggregation max`
(default) for the highest NOK score or `--binary-aggregation mean` for the arithmetic
average of the three NOK scores. The combined score at or above the cutoff means
NOK. Set `--nok-threshold 0.5` to override the checkpoint's saved cutoff; the accepted
range is 0 to 1 inclusive. All three matched views are always required.

`max` lets one high-scoring view trigger NOK; `mean` gives all three views equal
weight, which can dilute a defect visible from only one side. Compare both rules
and select their cutoffs on development bottles before the final recording.
The reason model also checks all three views and supplies independent findings;
these never override the binary verdict.

Both models use the same frozen MobileNet encoder, which converts each image into
features. Two small linear output layers read those features: one scores NOK and
the other scores the reasons. The loader verifies that both encoders match before
sharing that computation. It accepts the MobileNet `instructor_visual_ok_nok_v1`
and `gass_binary_experiment_v1` binary formats. For standalone crops, use
`classifier.py --checkpoint path/to/reasons.pt --binary-checkpoint path/to/binary.pt`
with the three existing image arguments.
Both commands accept `--binary-aggregation` and `--nok-threshold`.

For example, add either configuration to the same development-video command:

```bash
--binary-aggregation max --nok-threshold 0.5
--binary-aggregation mean --nok-threshold 0.5
```

JSON keeps `reason_status`, per-camera `binary_scores`, combined `binary_score`,
`binary_aggregation`, effective `binary_threshold` and `decision_source`
(`binary_max` or `binary_mean`). Run metadata and the video panel also show the
rule and effective cutoff. A binary NOK without a supported reason sets
`reason_unresolved`; different binary/reason verdicts set `reason_disagreement`.
The overlay shows these flags and any supporting findings, including on binary OKs.
This mode cannot use `--demo-dirt`. It is a development integration: the historical
binary checkpoint has no confirmed current project policy. Refit it from reviewed
`primagaz_only_v1` visual labels without reserved recording photos before claiming
current-policy or unseen-recording accuracy. Loading new reason weights does not
change the binary model's learned policy. OCR remains separate evidence.

This is preprocessed playback for integration, not a held-out accuracy evaluation
or a real-time speed claim. Matching uses crossing order and camera delays; unequal
counts stop classification, and inconsistent timings/unreadable crops produce
INCOMPLETE rather than OK. Counts include decisions completed during the clip.

References: [MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
and [fixed-feature transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

### Bottle-marking OCR (issues #20, #23, #35)

The OCR reader uses pretrained [EasyOCR](https://github.com/JaidedAI/EasyOCR) on
full-resolution bottle crops. It reads the whole crop and, with optional field
weights, reads each proposed field box. Whole-crop text can also supply a
candidate when its text box is centred inside a field proposal. Text elsewhere
remains available for review. Both passes read a grey, contrast-enhanced copy
and enlarge regions smaller than 1000 px by 2×. With field weights, the pretrained
[PARSeq](https://github.com/mindee/doctr) recognizer (docTR) also reads each padded
field box, unchanged, as one line. Each reading and candidate records its `engine`.

Use the field classes `tare_kg` and `approval_until_date`. The parser normalizes
`TW 13,4 kg` to `13.4` and keeps year, month and day precision distinct. It accepts
`2025`, `03/2025`, `2025/03`, `03/25`, and ISO dates such as `2025-03-01`. For
two-digit years, 50–99 map to 1950–1999 and 00–49 map to 2000–2049. Dates before
2000, such as old manufacture stamps, are not candidates (#37); a field with only
such dates is missing. Plausibility ranges, agreed with the team, also reject tare
values outside 5–25 kg and dates after 2040 (the 2025 recordings plus the longest
15-year approval), so absurd misreads cannot win. These parser rules never change
a human transcription. Malformed numbers and ambiguous day/month/year formats stay
in the raw text. Explicit net weight and litre markings are excluded from tare
candidates.

Selection uses the field detector confidence and the lowest OCR confidence among
tokens used to read the value. `ocr_selection.json` sets their relative weight
with `alpha` and defines per-field minimums for both scores and the winning
margin. After the two confidence checks, each distinct value ranks by its best
observation using `field_confidence**alpha * ocr_confidence**(1-alpha)`. Repeated
readings record support for a value. The score remains the best observation's
score. A value read by both EasyOCR and PARSeq, in any view of the bottle, is
`agreed`. It skips the OCR-confidence minimum but not the field minimum, and it
ranks before values that are not agreed. On development boxes EasyOCR was often
confidently wrong, while agreeing reads were always correct. The margin compares
the two highest distinct qualifying values of the same kind (agreed or not).

The supplied configuration uses `alpha = 0.5` and leaves the minimums unset.
It records `best` and `alternatives` for review and leaves the accepted `value`
as `null` until thresholds are configured. Choose thresholds on development
bottles, then evaluate once on held-out bottles. The ranking score is a heuristic.
Field proposals have a detector confidence floor of `0.01`.

To train a field detector, annotate development bottle crops in Label Studio
(#18) with one box per visible field, export in YOLO format to
`data/ocr_fields/` with a `data.yaml`, and train with the shared script. Keep test
bottles aside, and keep the weights in `src/runs/` until they are selected:

```bash
python src/scripts/model.py --mode full --data data/ocr_fields/data.yaml
```

Use a separate OCR environment so its headless OpenCV package stays separate
from the video preview's OpenCV. Field weights also need the base requirements.
Pass one or more crops. Use `--same-bottle` only for views of one known bottle.
Omit `--fields` for a whole-crop text baseline.

```bash
python3.13 -m venv src/runs/ocr/runtime
src/runs/ocr/runtime/bin/python -m pip install -r requirements-ocr.txt -r requirements.txt
src/runs/ocr/runtime/bin/python src/scripts/ocr.py data/prepared/ws5_14_55_v3/images/b001_front.jpg data/prepared/ws5_14_55_v3/images/b001_back_left.jpg data/prepared/ws5_14_55_v3/images/b001_back_right.jpg --same-bottle --fields src/runs/full/weights/best.pt --output src/runs/ocr/b001.json
```

On Windows, use `Scripts/python.exe` instead of `bin/python` in the environment.
The first invocation downloads public pretrained weights; images are processed
locally. CPU is the portable default; `--device mps` and `--device cuda:0` are
available. The JSON retains source paths, readings, proposals, candidates and
selections. Field and text coordinates are relative to the crop after EXIF
orientation. The previews show text boxes in yellow and field proposals in green.
Review both before setting thresholds. Existing output paths are refused.

Use `--selection-config` to try a copy of `src/scripts/ocr_selection.json` with
different thresholds. For saved standalone or three-camera results, rerun just
the selection with standard Python and write a fresh output file. Replay uses the
candidate scores saved by this version of the reader.

```bash
python src/scripts/ocr_selection.py src/runs/ocr/b001.json --config path/to/selection.json --output src/runs/ocr/b001-reselected.json
```

`evaluate_ocr.py` compares the combined bottle selections in a folder of bottle
caches (from the dataset packet's `run_ocr_groups.py`) with the verified workbook
references. With `--tune`, it replays the selection on validation bottles over a
threshold grid and keeps, per field, the fewest wrong acceptances and then the
most correct ones; ties take the middle setting. It writes
`selection.validation.json` and every trial. With `--config`, it reports a frozen
config, for example once on test bottles. Tare matches within 0.1 kg, declared
before tuning. Dates match on the approval-until year, the workbook's precision.
Each run saves per-bottle rows, metrics (correct, wrong, abstained, precision,
coverage and tare error in kg) and charts. Unknown references are not scored.

```bash
python src/scripts/evaluate_ocr.py src/runs/ocr-fields/v2-pilot/validation-cache --references data/ocr_v2/ground_truth/verified_reference_values.csv --output src/runs/ocr-fields/v2-pilot/validation-report --tune
```

`evaluate_recognition.py` measures recognition without the detector. It reads the
hand-drawn boxes that have a literal transcription, from the train and validation
splits only. Each box is read as drawn and with a character allowlist and/or
deskewing of rotated boxes. Results are exact text, character error rate and
whether the parser extracts the transcribed value. Run it from the OCR environment:

```bash
src/runs/ocr/runtime/bin/python src/scripts/evaluate_recognition.py data/ocr_v2 --output src/runs/ocr-fields/recognition
```

To use OCR in the three-camera pipeline, run it from the OCR environment and
add `--ocr`, plus `--ocr-fields` with field weights. Use `--ocr-selection-config`
to supply validation settings and `--no-display` with headless OpenCV.

```bash
src/runs/ocr/runtime/bin/python src/scripts/tricam_offline.py --ocr --ocr-fields src/runs/full/weights/best.pt --tracks-dir data/dataset/tracks --start-seconds 315 --seconds 5 --output src/runs/tricam_ocr/demo --device cpu --no-display
```

Add `--classifier` with a checkpoint to run visual inspection alongside OCR.
`inspect_crossings` builds each matched full-resolution camera crop once and
passes it to both readers before rendering. In `predictions.json`, each bottle
retains per-camera OCR evidence in `ocr` and combined selections in `ocr_fields`.
Winning observations record the camera, frame, track and both the bottle's frame
box and the field's crop box. The overlay shows the combined values and `?` for
unresolved fields. Tare and date can come from different camera views.

Visual status and counters describe the visual classifier (`status_scope` is
`visual`). Overall inspection rules are tracked in #25. With OCR alone, visual
status is `NOT_CHECKED` for usable matched triplets and counts are `null`.
Matching errors and unusable triplets are `INCOMPLETE`. Use development bottles
to improve OCR and preserve held-out footage for evaluation.

## Checks and GitHub workflow

```bash
python -m unittest discover -s tests -v
```

Tests cover device selection, invalid inputs, synthetic ByteTrack ID continuity,
classifier preprocessing, masked labels, split leakage, triplet decisions,
checkpoint loading, OCR candidate parsing, selection, provenance and replay, and
OCR evaluation (threshold tuning, scoring and rotated-box crops).
Real footage still needs checks for misses and ID switches.

Use **Issue → personal branch → focused commits → PR → teammate review → merge**.
Name branches `<issue-number>-<name>-<description>` and link the PR to its Issue.
Review code simplicity before pushing, then review the final PR diff before
merging. Require one teammate approval and resolve conversations. Verification
is manual; no CI checks are required. Do not push directly to `main` or force-push
shared branches. Update a clean local `main` with `git pull --ff-only` after merge.
Emilija's offered workflow guide can be incorporated after team review.

Data, generated outputs, and unselected weights are ignored. Historical local
`runs/` folders are preserved; new scripts write to `src/runs/`.
