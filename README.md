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

## Run

Place the dataset configuration at `data/data.yaml` and make sure its paths
point to your local dataset. Then, from the project root:

```bash
python src/model.py --mode quick
python src/model.py --mode full
```

Quick mode is the default: it trains on 20% of the training data for at most
0.4 hours (about 24 minutes), so a laptop CPU run finishes in under 30 minutes.
Ultralytics measures the first epoch and then trains as many epochs as fit in
that time. Full mode uses all training data for up to 50 epochs, stopping after
10 epochs without validation improvement; on a CPU this takes many hours, so use
Google Colab for it (see below). Both modes use 480 px images. Adjust the
constants at the top of `src/model.py` to change these settings.

The scripts automatically use the Mac GPU through MPS when available, then
CUDA on other supported computers, and otherwise the CPU. Use `--device cpu`
to force CPU execution.

If your dataset YAML is elsewhere, pass its location with `--data`, for example:

```bash
python src/model.py --mode quick --data data/my-dataset/data.yaml
```

Make sure the YAML's `train`, `val`, and `test` paths resolve to the correct
image folders. Keep the test split separate from training and validation.

Training prints the actual results directory and `best.pt` location. Repeated
runs get separate output directories. The pretrained weights download on the
first run; weights, datasets, and results stay out of Git. See the
[Ultralytics installation guide](https://docs.ultralytics.com/quickstart/)
for the underlying library.

## Train on Google Colab (recommended for the real model)

Colab gives you a free NVIDIA T4 GPU, which trains the full dataset far faster
than a laptop CPU. The notebook `colab/train_colab.ipynb` explains every step in
its own cells; this is the overview:

1. **Zip the dataset.** Make one zip with `data.yaml` and the `train`, `valid`
   and `test` folders directly inside it (no extra parent folder), and name it
   `cylinders_dataset.zip`. It is about 350 MB.
2. **Upload the zip to Google Drive.** Put it in the root of **My Drive**. If you
   use another folder, change `DATASET_ZIP` in cell 4 of the notebook.
3. **Open the notebook in Colab.** Go to
   [colab.research.google.com](https://colab.research.google.com), choose
   **File → Upload notebook** and select `colab/train_colab.ipynb` from this
   repository.
4. **Select the GPU.** Choose **Runtime → Change runtime type → T4 GPU** (the
   notebook asks for it by default, but check it).
5. **Run everything.** Choose **Runtime → Run all** and allow access to Google
   Drive when asked. The notebook:
   - checks that the GPU is there (`nvidia-smi`);
   - installs the same Ultralytics version as `requirements.txt`;
   - copies the zip from Drive to the Colab disk and unzips it into `/content/data`;
   - trains YOLOv8n on 100% of the training data at 480 px for at most 45 minutes
     (`time=0.75`), stopping earlier after 10 epochs without improvement;
   - saves the weights to `My Drive/gas_cylinders_runs/colab/weights/` after every
     epoch and shows `results.png` at the end.
6. **Keep the tab open.** Free Colab sessions disconnect when idle for too long
   or after a few hours. Nothing is lost because the weights are already on
   Drive; the last notebook cell shows how to resume from `last.pt`.
7. **Bring the model back.** Download `best.pt` from
   `My Drive/gas_cylinders_runs/colab/weights/` and put it in this repository as
   `runs/colab/weights/best.pt`. Then evaluate it and use it for tracking:

   ```bash
   python src/evaluate.py --weights runs/colab/weights/best.pt
   python src/track.py --weights runs/colab/weights/best.pt
   ```

To train for longer and get a better model, increase `time` in cell 5 (for
example `time=1.5` for 90 minutes).

## Evaluate

After training, use the printed best-weights path:

```bash
python src/evaluate.py --weights runs/full/weights/best.pt
```

Evaluation defaults to the held-out `test` split, which must be configured in
your dataset YAML. It reports precision, recall, mAP50, mAP50-95, and the
results location. Use `--data` for a different YAML location or `--split val`
to check the validation split; validation scores are not held-out test scores.
See the [Ultralytics validation guide](https://docs.ultralytics.com/modes/val/).

## Track

Tracking runs the detector on a video and links the detections from frame to
frame, so every cylinder keeps the same ID while it moves on the conveyor belt.
Put the video at `data/video_banda.mp4` (ignored by Git) and run:

```bash
python src/track.py --weights runs/colab/weights/best.pt
```

The script uses ByteTrack, keeps detections with at least 40% confidence and
processes 1 frame out of 3 to go faster. It saves the video with the boxes and
IDs under `runs/track/`. To try the BoT-SORT tracker, change `tracker` to
`"botsort.yaml"` in `src/track.py`. See the
[Ultralytics tracking guide](https://docs.ultralytics.com/modes/track/).
