# GASS AI Applications project
## Group name: The Global Variables
### Members: Isaac Smith, Sebastian Albu, Cosmin Brinda, Emilija Rimselite, Wood de Bock

We used CylinDeRS database to fine-tune a small yolo v8 nano model to recognize gas cylinders.

## Setup

Use Python 3.13. From the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.13 -m venv .venv` and
activate it with `.venv\Scripts\Activate.ps1` in PowerShell.

Select `.venv` as your Python interpreter in your IDE.

## Layout

```text
src/model.py              Training script
src/evaluate.py           Evaluation script
src/track.py              Tracking script (follows each cylinder through a video)
colab/train_colab.ipynb   Training notebook for Google Colab (free GPU)
data/                     Local dataset, data.yaml and videos (ignored by Git)
runs/                     Generated training results (ignored by Git)
requirements.txt          Python dependencies
```

## Dataset

Copy the CylinDeRS export (`data.yaml` plus the `train`, `valid` and `test`
folders) into `data/`, so the configuration is at `data/data.yaml`. The paths
in the Roboflow `data.yaml` (`../train/images`, ...) work as they are, because
Ultralytics also looks for them next to `data.yaml`. The whole `data/` folder is
ignored by Git, so the images are never committed or pushed.

## Train on your laptop (quick check)

From the project root:

```bash
python src/model.py
```

This trains on 20% of the training images for 0.3 hours (18 minutes).
Ultralytics measures the first epoch and then trains as many epochs as fit in
that time. The final validation runs after that, so on a laptop CPU the whole
run takes about 30 minutes. The weights are saved
to `runs/train/weights/best.pt`. The pretrained `yolov8n.pt` downloads on the
first run; weights, datasets and results stay out of Git.

## Train on Google Colab (recommended for the real model)

Colab gives you a free NVIDIA T4 GPU, which trains on all the images far faster
than a laptop CPU. The notebook `colab/train_colab.ipynb` explains every step in
its own cells:

1. **Zip the dataset.** Make one zip with `data.yaml` and the `train`, `valid`
   and `test` folders directly inside it (no extra parent folder), and name it
   `cylinders_dataset.zip`. It is about 350 MB.
2. **Upload the zip to Google Drive**, in the root of **My Drive**.
3. **Open the notebook in Colab.** Go to
   [colab.research.google.com](https://colab.research.google.com), choose
   **File → Upload notebook** and select `colab/train_colab.ipynb`.
4. **Select the GPU.** Check that **Runtime → Change runtime type** is set to
   **T4 GPU**.
5. **Run everything** with **Runtime → Run all** and allow access to Google
   Drive when asked. The notebook unzips the dataset, trains for 45 minutes
   (`time=0.75`) on all the training images and saves the weights to
   `My Drive/gas_cylinders_runs/colab/weights/` after every epoch.
6. **Keep the tab open.** Free Colab sessions disconnect when idle for too long.
   Nothing is lost because the weights are already on Drive; the last notebook
   cell shows how to resume from `last.pt`.
7. **Bring the model back.** Download `best.pt` from Drive and put it in this
   repository as `runs/colab/weights/best.pt`. In `src/evaluate.py` and
   `src/track.py`, change `"train"` to `"colab"` in `WEIGHTS`.

To get a better model, increase `time` in cell 5 (for example `time=1.5` for
90 minutes).

## Evaluate

```bash
python src/evaluate.py
```

This tests the weights from `WEIGHTS` on the `test` images, which the model
never saw during training, and prints precision, recall, mAP50 and mAP50-95.
See the [Ultralytics validation guide](https://docs.ultralytics.com/modes/val/).

## Track

Tracking runs the detector on a video and links the detections from frame to
frame, so every cylinder keeps the same ID while it moves on the conveyor belt.
Put the video in `data/` (ignored by Git), write its file name in `VIDEO` at the
top of `src/track.py` (now `filling_plant_bolivia.webm`) and run:

```bash
python src/track.py
```

The script uses ByteTrack, keeps detections with at least 40% confidence and
processes 1 frame out of 3 to go faster. It saves the video with the boxes and
IDs to `runs/track/`. To try the BoT-SORT tracker, change `tracker` to
`"botsort.yaml"` in `src/track.py`. See the
[Ultralytics tracking guide](https://docs.ultralytics.com/modes/track/).
