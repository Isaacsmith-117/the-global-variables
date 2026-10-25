# GASS: gas-cylinder detection and tracking

**The Global Variables:** Isaac Smith, Sebastian Albu, Cosmin Brinda,
Emilija Rimselite, and Wood de Bock.

Fine-tune YOLOv8 nano on CylinDeRS, evaluate the detector, and track cylinders
in local video using ByteTrack. A separate WS5 classifier checks matched bottle
crops for visible rejection reasons.

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

These are CylinDeRS detector scores. Conveyor tracking accuracy, ID stability,
and counting reliability remain unverified. Only the two selected models are
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

### Three-view classifier (WS5, issue #17)

The classifier takes three already-matched crops of one bottle: `front`,
`back_left`, and `back_right`. It returns one `OK`/`NOK`, triggered reasons,
and the camera views that supplied those reasons. `OK` means the enabled visual
checks passed; it does not certify every plant requirement. The offline video
integration below supplies matched crops and displays the results.

`classifier.py` uses frozen pretrained MobileNetV3-Small features and one small
trainable linear layer. Each crop is resized proportionally, padded to 224×224,
and normalized. Training caches features once, then trains only the linear layer
on CPU. `--device` controls feature extraction and prediction. This fixed-feature
approach is an efficient baseline; fine-tuning can be considered after evaluation.

The prepared folder contains `dataset.json` and a per-crop `manifest.jsonl`.
Targets are `0`, `1`, or `null`; masks exclude unknown and disabled labels.
Quarantine is excluded. Plant status is an evaluation reference, never a model
input. Every bottle's three views must belong to the same split.

```bash
python src/scripts/train_classifier.py train --data data/prepared/ws5_14_55_v3
python src/scripts/classifier.py --checkpoint src/runs/classifier/baseline/classifier.pt --front data/prepared/ws5_14_55_v3/images/b001_front.jpg --back-left data/prepared/ws5_14_55_v3/images/b001_back_left.jpg --back-right data/prepared/ws5_14_55_v3/images/b001_back_right.jpg --bottle-id 14_55:1
```

Training defaults to 20 epochs, batch size 32, AdamW at 0.001, and seed 0.
Positive-class weights come only from training labels. The best checkpoint is
chosen by validation loss. For each reason, take the maximum score over the three
views; validation bottles select a threshold in 0.05 steps to maximize F2, which
favours recall. Thresholds are saved with the checkpoint and never selected on test
bottles. These scores are not calibrated probabilities. The default output folder
must not already exist; use `--output src/runs/classifier/another-run` for a new run.
The first training invocation downloads public ImageNet weights; images stay local.

The current dataset enables `major_damage` and `wrong_brand_or_model`. Heavy dirt
is disabled because it has no positive examples. Training contains only one
damage-positive bottle and two model-positive bottles; validation has one of each.
Performance and threshold estimates are therefore exploratory. Inference reports
`checked_reasons` and `unchecked_reasons`, and rejects incomplete triplets.
NOK reasons are the supported label categories, not generated explanations.
API callers pass RGB PIL images or paths to `BottleClassifier.predict()`; convert
OpenCV BGR crops to RGB PIL images before calling it.

Evaluate only when the model and thresholds are settled:

```bash
python src/scripts/train_classifier.py evaluate --data data/prepared/ws5_14_55_v3 --checkpoint src/runs/classifier/baseline/classifier.pt
```

Evaluation reports per-reason TP/FN/FP/TN, precision/recall, unknown counts and plant
agreement per bottle. Recall is `null` when a reason has no positive test bottles,
as is currently the case for damage. Training opens only train/validation images;
test evaluation is a separate command. Checkpoints and validation reports stay in
ignored `src/runs/`; no classifier weights are selected for team use yet.

### Annotated three-camera classifier video

`tricam_offline.py --classifier` connects the classifier to the existing 2×2
camera display. Video order is **back-left, back-right, front**. Each bottle is
classified once using its three line-crossing crops. All crops come from the
videos and tracker boxes; plant labels are not used to generate predictions.
The display shows grey Inspecting boxes, green OK boxes, red NOK boxes with
reasons, and a panel counting completed inspections once per bottle. Heavy dirt
remains marked unchecked. Results appear after all three crossing crops are
available in the synchronized playback.

```bash
python src/scripts/tricam_offline.py --classifier src/runs/classifier/smoke-20261025/classifier.pt --tracks-dir data/dataset/tracks --start-seconds 315 --seconds 45 --output src/runs/tricam_classifier/demo --device cpu --no-display
```

Without `--no-display`, a preview window is shown while rendering; press Q to stop.
The output folder receives `output.mp4` and `predictions.json` with decisions,
scores, evidence views, crop frames/boxes and clip counts. Existing output files
are never overwritten. Omit `--tracks-dir` to run the detector and create tracks;
use `--videos` to provide three other recordings. Omit `--seconds` for the full
synchronized recording. Saved tracks must correspond to the supplied videos.

For a dirt-enabled integration demo, add `--demo-dirt`. With the current two-reason
checkpoint this includes its existing untrained dirt output at a fixed 0.5 cutoff;
it is not evidence of dirt-detection performance. The overlay uses the normal
`heavy dirt` reason text. Prediction JSON records `demo_only_reasons` and the run
records `demo_dirt_requested`. The checkpoint, dataset and normal default behaviour
are unchanged. A future checkpoint with trained dirt keeps its saved threshold.

This is preprocessed playback for integration, not a held-out accuracy evaluation
or a real-time speed claim. Matching uses crossing order and camera delays; unequal
counts stop classification, and inconsistent timings/unreadable crops produce
INCOMPLETE rather than OK. Counts include decisions completed during the clip.

References: [MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
and [fixed-feature transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

### Bottle-marking OCR (issue #20)

The initial OCR reader uses pretrained [EasyOCR](https://github.com/JaidedAI/EasyOCR)
on full-resolution bottle crops. It finds text regions and reads their characters;
a small parser then collects explicit `kg` values and standalone four-digit years.
For example, `13,4 kg` produces a weight candidate of `13.4`. These are candidate
markings: a weight may be a capacity rather than tare, and a year may refer to
manufacture rather than certification. Missing candidates remain empty lists.
Raw text, confidence and text boxes are retained for review; confidence scores
are not calibrated probabilities. Two-digit years, split text regions and unclear
characters are left for later improvement.

A three-view development smoke run produced text but no verified tare and
certification-year pair. Recognition accuracy and processing speed have not
been established; this is an initial integration for further improvement.

Use a separate OCR environment so its headless OpenCV package stays separate
from the video preview's OpenCV. Then pass one or more crops:

```bash
python3.13 -m venv src/runs/ocr/runtime
src/runs/ocr/runtime/bin/python -m pip install -r requirements-ocr.txt
src/runs/ocr/runtime/bin/python src/scripts/ocr.py data/prepared/ws5_14_55_v3/images/b001_front.jpg data/prepared/ws5_14_55_v3/images/b001_back_left.jpg data/prepared/ws5_14_55_v3/images/b001_back_right.jpg --output src/runs/ocr/b001.json
```

On Windows, use `Scripts/python.exe` instead of `bin/python` in the environment.
The first invocation downloads public pretrained weights; images are processed
locally. CPU is the portable default; `--device mps` and `--device cuda:0` are
available. The JSON file records each source path and its text readings, with
box coordinates relative to the crop after EXIF orientation. Existing output
files are refused. Review the readings against the images before trusting fields.

To use OCR in the three-camera pipeline, also install the base requirements in
the OCR environment, then add `--ocr`. Use `--no-display` with headless OpenCV:

```bash
src/runs/ocr/runtime/bin/python -m pip install -r requirements.txt
src/runs/ocr/runtime/bin/python src/scripts/tricam_offline.py --ocr --tracks-dir data/dataset/tracks --start-seconds 315 --seconds 5 --output src/runs/tricam_ocr/demo --device cpu --no-display
```

Add `--classifier` with a checkpoint to run visual inspection alongside OCR.
`inspect_crossings` builds each matched full-resolution camera crop once and
passes it to both readers. OCR runs once per view/bottle before rendering, not
on every playback frame. Each bottle in `predictions.json` retains per-camera
OCR readings and candidate fields, alongside frame/track/bottle-box provenance.
OCR text boxes use coordinates within the corresponding oriented bottle crop.
The overlay shows candidate weights/years; `?` means none were extracted, not
that the physical marking is absent. Crossing images may not expose the text.

Visual status and counters describe the visual classifier only (`status_scope`
is `visual`). OCR candidates do not change OK/NOK; an overall acceptance rule is
still pending. With OCR alone, visual status is `NOT_CHECKED` for usable matched
triplets and visual counts are `null`. Matching errors/unusable triplets remain
`INCOMPLETE`, and neither reader processes them. The standalone reader does not
infer identity from filenames. For improvements, use training/development
bottles and keep test bottles aside; a video clip may contain test bottles too.

## Checks and GitHub workflow

```bash
python -m unittest discover -s tests -v
```

Tests cover device selection, invalid inputs, synthetic ByteTrack ID continuity,
classifier preprocessing, masked labels, split leakage, triplet decisions and
checkpoint loading. Real footage still needs checks for misses and ID switches.

Use **Issue → personal branch → focused commits → PR → teammate review → merge**.
Name branches `<issue-number>-<name>-<description>` and link the PR to its Issue.
Review code simplicity before pushing, then review the final PR diff before
merging. Require one teammate approval and resolve conversations. Verification
is manual; no CI checks are required. Do not push directly to `main` or force-push
shared branches. Update a clean local `main` with `git pull --ff-only` after merge.
Emilija's offered workflow guide can be incorporated after team review.

Data, generated outputs, and unselected weights are ignored. Historical local
`runs/` folders are preserved; new scripts write to `src/runs/`.
