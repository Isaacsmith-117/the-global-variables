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
src/model.py       Training script
src/evaluate.py    Evaluation script
data/              Local dataset and data.yaml (ignored by Git)
runs/              Generated training results (ignored by Git)
requirements.txt   Python dependencies
```

## Run

Place the dataset configuration at `data/data.yaml` and make sure its paths
point to your local dataset. Then, from the project root:

```bash
python src/model.py --mode quick
python src/model.py --mode full
```

Quick mode is the default: 3 epochs using 10% of the training data. Full mode
uses all training data for up to 50 epochs, stopping after 10 epochs without
validation improvement. Adjust the constants at the top of `src/model.py` to
change these settings.

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
