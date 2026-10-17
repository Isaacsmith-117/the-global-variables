# GASS: gas-cylinder detection and tracking

**The Global Variables:** Isaac Smith, Sebastian Albu, Cosmin Brinda,
Emilija Rimselite, and Wood de Bock.

Fine-tune YOLOv8 nano on CylinDeRS, evaluate the detector, and track cylinders
in local video using ByteTrack.

## Layout

```text
src/scripts/       Training, evaluation, tracking, and shared config
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

### Synchronized three-camera tracking

Create the folder `data/video/` (ignored by Git) and put the camera videos in
it. There are two scripts; both take the three videos with `--videos`, and
without it they use `14_55_back_left_cropped.mp4`, `14_55_back_right_cropped.mp4`
and `14_55_front_cropped.mp4` from `data/video/`. Each part of the screen shows
the name of its video file.

#### Live: `tricam.py`

```bash
python src/scripts/tricam.py --overwrite --videos data/video/first.mp4 data/video/second.mp4 data/video/third.mp4
```

The script reads one frame of every camera at each step, so the views stay
synchronized and you watch the tracking in real time; it stops when the
shortest video ends. Every camera has its own ByteTrack tracker; a shared ID
links the same bottle across the three views by comparing colour histograms,
but only with bottles seen in the last `MATCH_RECENT` frames, so a new bottle
does not take an old ID. A bottle is counted and gets its number only after it
was seen in `MIN_FRAMES` frames, so a detection that flickers for a moment is
not counted. The window and `src/runs/tricam/output.mp4` are split in four: the
three annotated cameras and a panel with the bottles seen so far and the
bottles in view now. Use `--no-display` to only save the video and `--no-save`
to only watch it; press `q` to stop. Matching by colour can mix up bottles that
look alike, so the counts still need manual review on real footage.

#### Offline, Emilija's method: `tricam_offline.py`

```bash
python src/scripts/tricam_offline.py --videos data/video/first.mp4 data/video/second.mp4 data/video/third.mp4
```

This script first tracks the whole videos, so it is not live, but the bottles
keep the same ID even though they all look alike:

1. Every video is tracked with ByteTrack and its boxes are saved to
   `src/runs/tricam_offline/<video name>.csv`. The next run reuses these files;
   delete them to track again, for example after changing the weights.
2. Only the real bottles are kept: tracks of at least `MIN_FRAMES` frames that
   move across at least `MIN_MOVE_FRAC` of the image width.
3. The bottles are numbered in the order they cross a line in the middle of the
   image (`LINE_FRAC`): the n-th bottle that crosses the line gets ID n in
   every camera.
4. The delay of each camera is the median difference between the frames in
   which the same bottle crosses the line. The videos are aligned by this delay
   and saved to `src/runs/tricam_offline/output.mp4`, split in four: the three annotated
   videos and a panel with the bottles seen so far and the bottles in view now.

The script prints the number of bottles and the delay of every camera. If the
cameras counted a different number of bottles, or many bottles have an unusual
delay, the IDs can be shifted between the cameras. Use `--no-display` to only
save the video; press `q` to stop.

## Checks and GitHub workflow

```bash
python -m unittest discover -s tests -v
```

Tests cover device selection, invalid inputs, and synthetic ByteTrack ID
continuity. Real footage still needs manual review for misses and ID switches.

Use **Issue → personal branch → focused commits → PR → teammate review → merge**.
Name branches `<issue-number>-<name>-<description>` and link the PR to its Issue.
Review code simplicity before pushing, then review the final PR diff before
merging. Require one teammate approval and resolve conversations. Verification
is manual; no CI checks are required. Do not push directly to `main` or force-push
shared branches. Update a clean local `main` with `git pull --ff-only` after merge.
Emilija's offered workflow guide can be incorporated after team review.

Data, generated outputs, and unselected weights are ignored. Historical local
`runs/` folders are preserved; new scripts write to `src/runs/`.
