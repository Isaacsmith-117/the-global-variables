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
data/              Local dataset and data.yaml (ignored by Git)
runs/              Generated training results (ignored by Git)
requirements.txt   Python dependencies
```

## Run

Place the dataset configuration at `data/data.yaml` and make sure its paths
point to your local dataset. Then, from the project root:

```bash
python src/model.py
```

The script runs the existing short training check (3 epochs, 10% of the
training data). The pretrained weights download on the first run; weights,
datasets, and results stay out of Git. See the
[Ultralytics installation guide](https://docs.ultralytics.com/quickstart/)
for the underlying library.
