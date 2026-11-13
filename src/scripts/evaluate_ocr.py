"""Choose OCR thresholds on validation bottles, then report bottle-level agreement.

Compares the combined (--same-bottle) selections in a folder of bottle caches with the
verified workbook references and saves per-bottle rows, metrics and charts:

python src/scripts/evaluate_ocr.py <validation-cache> --references <csv> --output <dir> --tune
python src/scripts/evaluate_ocr.py <test-cache> --references <csv> --output <dir> --config <json>
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__:
    from .ocr_selection import load_selection, select_fields
else:
    from ocr_selection import load_selection, select_fields

FIELDS = ("tare_kg", "approval_until_date")
TARE_TOLERANCE_KG = 0.1  # declared before tuning: stamps show the tare to 0.1 kg
GRID = [round(0.05 * step, 2) for step in range(20)]  # confidence thresholds 0.00-0.95
MARGINS = [0.0, 0.05, 0.1, 0.2]
COLORS = {"correct": "#2e7d32", "wrong": "#c62828", "abstained": "#9e9e9e", "no_reference": "#d7ccc8"}


def load_references(path):
    """Bottle-level references; values not marked eligible stay unknown and are not scored."""
    references = {}
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            reference = {
                "tare_kg": float(row["workbook_tarra_kg"]) if row["tare_reference_eligible"] == "True" else None,
                "approval_until_date": (int(float(row["approval_until_year"]))
                                        if row["approval_until_reference_eligible"] == "True" else None),
            }
            if references.setdefault(row["bottle_group"], reference) != reference:
                raise SystemExit(f"Views of {row['bottle_group']} have different references.")
    return references


def load_caches(folder):
    """One OCR cache per bottle group, as written by run_ocr_groups.py."""
    documents = {}
    for path in sorted(Path(folder).glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if "images" in document:  # skips run_receipt.json
            documents[document["dataset_split_provenance"]["bottle_group"]] = document
    return documents


def matches(field, value, reference):
    if field == "tare_kg":
        return abs(value - reference) <= TARE_TOLERANCE_KG + 1e-9
    return int(str(value)[:4]) == reference  # the workbook gives the approval year only


def score(documents, references, config):
    """One row per bottle and field: the selection outcome against the reference."""
    rows = []
    for group, document in sorted(documents.items()):
        selected = select_fields({view["image_path"]: view for view in document["images"]}, config)
        for field in FIELDS:
            chosen, reference, best = selected[field], references[group][field], selected[field]["best"]
            best_value = best["value"] if best else None
            if reference is None:
                outcome = "no_reference"
            elif chosen["status"] != "accepted":
                outcome = "abstained"
            else:
                outcome = "correct" if matches(field, chosen["value"], reference) else "wrong"
            known = reference is not None and best_value is not None
            rows.append({
                "bottle_group": group, "field": field, "reference": reference, "status": chosen["status"],
                "accepted_value": chosen["value"], "outcome": outcome, "best_value": best_value,
                "best_correct": matches(field, best_value, reference) if known else None,
                "best_field_confidence": best["field_confidence"] if best else None,
                "best_ocr_confidence": best["ocr_confidence"] if best else None,
                "best_view": best["view"] if best else None,
                "tare_error_kg": round(best_value - reference, 3) if known and field == "tare_kg" else None,
            })
    return rows


def metrics(rows):
    """Counts per field; precision is undefined (null) when nothing was accepted."""
    summary = {}
    for field in FIELDS:
        field_rows = [row for row in rows if row["field"] == field]
        counts = {kind: sum(row["outcome"] == kind for row in field_rows) for kind in COLORS}
        accepted = counts["correct"] + counts["wrong"]
        eligible = accepted + counts["abstained"]
        summary[field] = {
            **counts, "bottles": len(field_rows), "eligible": eligible,
            "accepted_precision": counts["correct"] / accepted if accepted else None,
            "coverage": accepted / eligible if eligible else None,
            "accepted_without_reference": sum(row["outcome"] == "no_reference" and row["status"] == "accepted"
                                              for row in field_rows),
            "best_read_correct": sum(row["best_correct"] is True for row in field_rows),
        }
        if field == "tare_kg":
            errors = [abs(row["tare_error_kg"]) for row in field_rows if row["tare_error_kg"] is not None]
            accepted_errors = [abs(row["tare_error_kg"]) for row in field_rows
                               if row["tare_error_kg"] is not None and row["outcome"] in ("correct", "wrong")]
            summary[field].update(
                best_read_mean_abs_error_kg=round(fmean(errors), 3) if errors else None,
                best_read_max_abs_error_kg=max(errors, default=None),
                accepted_mean_abs_error_kg=round(fmean(accepted_errors), 3) if accepted_errors else None)
    return summary


def tune(documents, references, base):
    """Per field: fewest wrong acceptances, then most correct; ties take the middle setting."""
    config = json.loads(json.dumps(base))
    trials = []
    for field in FIELDS:
        results = []
        for field_min in GRID:
            for ocr_min in GRID:
                for margin in MARGINS:
                    limits = {"min_field_confidence": field_min, "min_ocr_confidence": ocr_min,
                              "min_margin": margin}
                    trial = {**config, "fields": {**config["fields"], field: limits}}
                    outcomes = [row["outcome"] for row in score(documents, references, trial)
                                if row["field"] == field]
                    counts = {kind: outcomes.count(kind) for kind in ("correct", "wrong", "abstained")}
                    trials.append({"field": field, **limits, **counts})
                    results.append((counts["wrong"], -counts["correct"], field_min + ocr_min, margin, limits))
        best = min(result[:2] for result in results)
        tied = sorted((result for result in results if result[:2] == best), key=lambda result: result[2:4])
        config["fields"][field] = tied[len(tied) // 2][4]
    return config, trials


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def charts(rows, config, output):
    """Outcome counts, tare error per bottle, and confidences of the best reads."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bottom = [0] * len(FIELDS)
    for kind, color in COLORS.items():
        values = [sum(row["field"] == field and row["outcome"] == kind for row in rows) for field in FIELDS]
        ax.bar(FIELDS, values, bottom=bottom, color=color, label=kind.replace("_", " "))
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set(ylabel="bottles", title="OCR outcome per field")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "outcomes.png", dpi=150)
    plt.close(fig)

    tare = [row for row in rows if row["tare_error_kg"] is not None]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axhspan(-TARE_TOLERANCE_KG, TARE_TOLERANCE_KG, color="#c8e6c9", label=f"±{TARE_TOLERANCE_KG} kg tolerance")
    ax.bar([row["bottle_group"] for row in tare], [row["tare_error_kg"] for row in tare],
           color=[COLORS[row["outcome"]] for row in tare])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="best tare read − reference (kg)", title="Tare error per bottle (colour = outcome)")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "tare_error.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(FIELDS), figsize=(9, 4))
    for ax, field in zip(axes, FIELDS):
        for correct, kind in ((True, "correct"), (False, "wrong")):
            points = [row for row in rows if row["field"] == field and row["best_correct"] is correct]
            ax.scatter([row["best_field_confidence"] for row in points], [row["best_ocr_confidence"] for row in points],
                       color=COLORS[kind], label=f"best read {kind}")
        limits = config["fields"][field]
        if limits["min_field_confidence"] is not None:
            ax.axvline(limits["min_field_confidence"], color="black", linestyle="--", linewidth=0.8)
            ax.axhline(limits["min_ocr_confidence"], color="black", linestyle="--", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="field-box confidence", ylabel="OCR confidence", title=field)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "confidence.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", type=Path, help="Folder of bottle caches from run_ocr_groups.py.")
    parser.add_argument("--references", type=Path, required=True, help="verified_reference_values.csv")
    parser.add_argument("--output", type=Path, required=True, help="New folder for the report.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--tune", action="store_true", help="Choose thresholds on these validation bottles.")
    choice.add_argument("--config", type=Path, help="Frozen selection config to report with.")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new --output folder.")
    documents, references = load_caches(args.cache), load_references(args.references)
    if not documents or set(documents) - set(references):
        raise SystemExit("Every bottle cache needs a row in the references file.")

    args.output.mkdir(parents=True)
    if args.tune:
        config, trials = tune(documents, references, next(iter(documents.values()))["ocr_selection"])
        (args.output / "selection.validation.json").write_text(json.dumps(config, indent=2) + "\n")
        write_csv(args.output / "tuning_trials.csv", trials)
    else:
        config = load_selection(args.config)
    rows = score(documents, references, config)
    write_csv(args.output / "bottles.csv", rows)
    report = {"bottles": len(documents), "tare_tolerance_kg": TARE_TOLERANCE_KG,
              "date_comparison": "accepted year equals the workbook approval-until year",
              "selection": config, "metrics": metrics(rows)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    charts(rows, config, args.output)
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved the report in {args.output}")


if __name__ == "__main__":
    main()
