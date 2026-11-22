"""Link modular-pipeline bottle results to a ground-truth Excel workbook."""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from openpyxl.styles import PatternFill
except ImportError as error:
    raise SystemExit("Install openpyxl to export results.xlsx.") from error


def _clean_key(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(number):
        if float(number).is_integer():
            return str(int(number))
        return str(number)
    return re.sub(r"[^a-z0-9]+", "", text.lower()) or None


def _find_column(frame, names):
    normalized = {re.sub(r"[^a-z0-9]+", "", str(column).lower()): column
                  for column in frame.columns}
    for name in names:
        key = re.sub(r"[^a-z0-9]+", "", name.lower())
        if key in normalized:
            return normalized[key]
    return None


def _read_run_metadata(bottles_path):
    metadata_path = bottles_path.parent / "predictions.json"
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _prediction_timestamp(row, prediction_frame, fps, start_frame):
    elapsed_columns = (
        "timestamp_seconds", "time_seconds", "ready_time_seconds",
        "prediction_time_seconds",
    )
    for column in elapsed_columns:
        found = _find_column(pd.DataFrame([row]), (column,))
        if found is not None:
            numeric = pd.to_numeric(pd.Series([row[found]]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                return float(numeric), "elapsed"

    for column in ("timestamp", "Time"):
        found = _find_column(pd.DataFrame([row]), (column,))
        if found is not None:
            value = row[found]
            if pd.notna(value):
                parsed = pd.to_datetime(value, errors="coerce")
                if pd.notna(parsed):
                    return parsed.to_pydatetime(), "absolute"

    if prediction_frame is not None:
        value = row.get(prediction_frame)
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if fps is not None and pd.notna(numeric):
            frame_offset = start_frame or 0
            return float(numeric - frame_offset) / float(fps), "elapsed"
    return None


def _ground_truth_timestamps(frame, gt_time_column):
    if gt_time_column is None:
        return None
    parsed = pd.to_datetime(frame[gt_time_column], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return pd.Series(pd.NaT, index=frame.index), None
    origin = valid.iloc[0]
    seconds = (parsed - origin).dt.total_seconds()
    return seconds, origin


def evaluate(ground_truth_path, bottles_path, output_path, fps=None,
             time_tolerance=3.0, time_offset=0.0):
    ground_truth = pd.read_excel(ground_truth_path)
    predictions = pd.read_csv(bottles_path)

    if predictions.empty:
        result = ground_truth.copy()
    else:
        gt_id = _find_column(ground_truth, ("id", "bottle_id", "bottle id"))
        gt_sequence = _find_column(ground_truth, ("Sequence", "sequence", "cylinder_sequence"))
        gt_time = _find_column(ground_truth, ("Time", "timestamp", "time"))

        prediction_id = _find_column(predictions, ("bottle_id", "id", "bottle id"))
        prediction_sequence = _find_column(
            predictions, ("Sequence", "sequence", "bottle_sequence", "cylinder_sequence"))
        prediction_frame = _find_column(predictions, ("ready_frame", "frame", "ready frame"))

        metadata = _read_run_metadata(bottles_path)
        start_frame = None
        if metadata:
            fps = fps or metadata.get("fps")
            start_frame = metadata.get("start_frame")

        gt_seconds, gt_origin = _ground_truth_timestamps(ground_truth, gt_time)

        output_rows = []
        used_predictions = set()

        for gt_index, gt_row in ground_truth.iterrows():
            match_index = None

            if gt_sequence is not None and prediction_sequence is not None:
                gt_sequence_key = _clean_key(gt_row.get(gt_sequence))
                if gt_sequence_key:
                    for prediction_index, prediction_row in predictions.iterrows():
                        if prediction_index in used_predictions:
                            continue
                        if _clean_key(prediction_row.get(prediction_sequence)) == gt_sequence_key:
                            match_index = prediction_index
                            break

            if match_index is None and gt_id is not None and prediction_id is not None:
                gt_id_key = _clean_key(gt_row.get(gt_id))
                if gt_id_key:
                    for prediction_index, prediction_row in predictions.iterrows():
                        if prediction_index in used_predictions:
                            continue
                        if _clean_key(prediction_row.get(prediction_id)) == gt_id_key:
                            match_index = prediction_index
                            break

            if match_index is None and gt_seconds is not None:
                gt_seconds_value = gt_seconds.loc[gt_index]
                if pd.notna(gt_seconds_value):
                    gt_elapsed = float(gt_seconds_value) + float(time_offset)
                    best_index = None
                    best_difference = float("inf")
                    for prediction_index, prediction_row in predictions.iterrows():
                        if prediction_index in used_predictions:
                            continue
                        timestamp = _prediction_timestamp(
                            prediction_row, prediction_frame, fps, start_frame)
                        if timestamp is None:
                            continue
                        timestamp_value, timestamp_kind = timestamp
                        if timestamp_kind == "absolute":
                            if gt_origin is None:
                                continue
                            prediction_elapsed = (
                                pd.Timestamp(timestamp_value) - gt_origin
                            ).total_seconds()
                        else:
                            prediction_elapsed = float(timestamp_value)
                        difference = abs(prediction_elapsed - gt_elapsed)
                        if difference < best_difference:
                            best_difference = difference
                            best_index = prediction_index
                    if best_index is not None and best_difference <= float(time_tolerance):
                        match_index = best_index

            combined = gt_row.to_dict()
            if match_index is not None:
                used_predictions.add(match_index)
                for column, value in predictions.loc[match_index].items():
                    combined[f"{column}_prediction"] = value
            output_rows.append(combined)

        result = pd.DataFrame(output_rows)

        remaining = predictions.loc[~predictions.index.isin(used_predictions)]
        if not remaining.empty:
            remaining_rows = []
            for _, prediction_row in remaining.iterrows():
                row = {column: pd.NA for column in ground_truth.columns}
                for column, value in prediction_row.items():
                    row[f"{column}_prediction"] = value
                remaining_rows.append(row)

            result = pd.concat(
                [result, pd.DataFrame(remaining_rows)], ignore_index=True, sort=False
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Results")
        worksheet = writer.sheets["Results"]

        prediction_columns = [
            column for column in result.columns
            if str(column).endswith("_prediction")
        ]
        green_fill = PatternFill(fill_type="solid", fgColor="C6EFCE")

        for cell in worksheet[1]:
            if cell.value in prediction_columns:
                cell.fill = green_fill
        for column_index, column_name in enumerate(result.columns, start=1):
            if column_name in prediction_columns:
                for row_index in range(2, worksheet.max_row + 1):
                    worksheet.cell(row=row_index, column=column_index).fill = green_fill

                values = [str(value) for value in result[column_name].dropna().tolist()]
                max_length = max([len(str(column_name)), *[len(value) for value in values]])
                worksheet.column_dimensions[worksheet.cell(row=1, column=column_index).column_letter].width = min(max_length + 2, 45)

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_index, column_name in enumerate(result.columns, start=1):
            if column_name not in prediction_columns:
                letter = worksheet.cell(row=1, column=column_index).column_letter
                values = [str(value) for value in result[column_name].dropna().head(200).tolist()]
                max_length = max([len(str(column_name)), *[len(value) for value in values]])
                worksheet.column_dimensions[letter].width = min(max_length + 2, 45)

    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Link bottle.csv predictions to a ground-truth workbook."
    )
    parser.add_argument("--bottles", "-b", required=True, type=Path,
                        help="Path to bottles.csv")
    parser.add_argument("--ground-truth", "-g", required=True, type=Path,
                        help="Path to the ground-truth XLSX")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("results.xlsx"), help="Output XLSX")
    parser.add_argument("--fps", type=float,
                        help="FPS for ready_frame timestamp matching; defaults to sibling predictions.json")
    parser.add_argument("--time-tolerance", type=float, default=3.0,
                        help="Maximum seconds between GT and prediction timestamps")
    parser.add_argument("--time-offset", type=float, default=0.0,
                        help="Prediction-time offset in seconds relative to the first ground-truth timestamp")
    args = parser.parse_args(argv)

    if args.time_tolerance < 0:
        parser.error("--time-tolerance must be non-negative")

    if not args.bottles.is_file():
        parser.error(f"Bottle CSV not found: {args.bottles}")
    if not args.ground_truth.is_file():
        parser.error(f"Ground-truth workbook not found: {args.ground_truth}")

    output = evaluate(
        args.ground_truth,
        args.bottles,
        args.output,
        fps=args.fps,
        time_tolerance=args.time_tolerance,
        time_offset=args.time_offset,
    )
    print(f"Results: {output}")
    return output


if __name__ == "__main__":
    main()
