"""Parse and rank OCR evidence; replay saved results without loading either model."""

import argparse
from datetime import date
import json
from pathlib import Path
import re

CONFIG_PATH = Path(__file__).with_suffix(".json")
NUMERIC = re.compile(r"[0-9]+(?:\s*[.,/-]\s*[0-9]+)*")
NUMBER_EDGES = tuple(".,/+-")
TARE_LABELS = {"t", "tw", "tara", "tarra", "tare"}
WEIGHT_LABEL = re.compile(r"\b(?:tw|tara|tarra|tare|netto|net|butaan|propaan|butane|propane)\b", re.I)
LIMITS = {"min_field_confidence", "min_ocr_confidence", "min_margin"}


def parse_candidates(field, text):
    """Keep whole numeric spans; never recover a valid fragment of malformed text."""
    candidates = []
    previous_end = 0
    for match in NUMERIC.finditer(text):
        before, after = text[:match.start()], text[match.end():]
        context = text[previous_end:match.start()]
        previous_end = match.end()
        if before.rstrip().endswith(NUMBER_EDGES) or after.lstrip().startswith(NUMBER_EDGES):
            continue
        raw = re.sub(r"\s+", "", match.group())
        value = None
        if field.endswith("_kg") and re.fullmatch(r"\d{1,2}(?:[.,]\d{1,2})?", raw):
            attached_label = re.search(r"[a-z]+$", before, re.I)
            if after[:1].isalpha() and not re.match(r"kg\b", after, re.I):
                continue
            labels = WEIGHT_LABEL.findall(context)
            label = attached_label.group().lower() if attached_label else (labels[-1].lower() if labels else None)
            if label and label not in TARE_LABELS:
                continue
            if re.match(r"\s*(?:l\b|lit(?:er|re))", after, re.I):
                continue
            if not label and re.match(r"\s*(?:kg\s+)?net(?:to)?\b", after, re.I):
                continue
            value = float(raw.replace(",", ".")) or None
        elif field.endswith("_date"):
            if before[-1:].isalnum() or after[:1].isalnum() or re.search(r"\bun\s*$", before, re.I):
                continue  # e.g. UN1965 and serial numbers
            parts = re.split(r"[/.-]", raw)
            try:
                if len(parts) == 1 and re.fullmatch(r"(?:19|20)\d{2}", raw):
                    value = raw
                elif len(parts) == 2:
                    year, month = parts if len(parts[0]) == 4 else parts[::-1]
                    if len(year) in (2, 4) and 1 <= len(month) <= 2:
                        year = int(year)
                        if year < 100:
                            year += 1900 if year >= 50 else 2000
                        date(year, int(month), 1)
                        if 1900 <= year <= 2099:
                            value = f"{year}-{int(month):02d}"
                elif re.fullmatch(r"(?:19|20)[0-9]{2}-[0-9]{2}-[0-9]{2}", raw):
                    value = date.fromisoformat(raw).isoformat()
            except ValueError:
                pass
        if value is not None:
            candidates.append({"value": value, "start": match.start(), "end": match.end(),
                               "value_text": match.group()})
    return candidates


def parse_value(field, text):
    """Compatibility helper: only return an unambiguous normalized value."""
    values = {candidate["value"] for candidate in parse_candidates(field, text)}
    return next(iter(values)) if len(values) == 1 else None


def unit_score(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and 0 <= value <= 1)


def load_selection(path=None):
    config = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"alpha", "fields"} or not unit_score(config["alpha"]):
        raise ValueError("Selection config needs alpha in [0, 1] and fields.")
    if not isinstance(config["fields"], dict) or not config["fields"]:
        raise ValueError("Selection config needs at least one field.")
    for name, limits in config["fields"].items():
        if (not name.endswith(("_kg", "_date")) or not isinstance(limits, dict)
                or set(limits) != LIMITS
                or any(value is not None and not unit_score(value) for value in limits.values())):
            raise ValueError(f"Invalid selection limits for {name}.")
    return config


def select_fields(views, config, provenance=None):
    """Gate observations, group equal values, then rank by the best eligible observation.

    Scores are ranking heuristics, not probabilities. Repeated reads get no vote bonus.
    Provenance is supplied by the caller that has already matched the bottle views.
    """
    selected = {}
    for field, limits in config["fields"].items():
        configured = all(value is not None for value in limits.values())
        groups = {}
        for view_name, view in views.items():
            for candidate in view["candidates"]:
                if candidate["field"] != field or candidate["value"] is None:
                    continue
                field_score, text_score = candidate["field_confidence"], candidate["ocr_confidence"]
                if any(score is not None and not unit_score(score) for score in (field_score, text_score)):
                    raise ValueError("Candidate confidence must be finite and in [0, 1].")
                scored = field_score is not None and text_score is not None
                score = field_score ** config["alpha"] * text_score ** (1 - config["alpha"]) if scored else None
                qualified = (configured and scored and field_score >= limits["min_field_confidence"]
                             and text_score >= limits["min_ocr_confidence"])
                observation = {**candidate, "view": view_name, "score": score, "qualified": qualified}
                if provenance is not None:
                    crop = provenance[view_name]
                    observation.update(camera=view_name, frame=crop["frame"], track_id=crop["track_id"],
                                       bottle_box=crop["box"])
                if "image_path" in view:
                    observation["image_path"] = view["image_path"]
                groups.setdefault(candidate["value"], []).append(observation)
        alternatives = []
        for value, observations in groups.items():
            observations.sort(key=lambda c: (c["qualified"], c["score"] if c["score"] is not None else -1), reverse=True)
            best = observations[0]
            alternatives.append({"value": value, "score": best["score"], "qualified": best["qualified"],
                                 "support_views": sorted({c["view"] for c in observations}),
                                 "observations": observations})
        alternatives.sort(key=lambda c: (c["qualified"], c["score"] if c["score"] is not None else -1), reverse=True)
        eligible = [c for c in alternatives if c["qualified"]]
        margin = eligible[0]["score"] - eligible[1]["score"] if len(eligible) > 1 else None
        if not alternatives:
            status = "missing"
        elif not configured:
            status = "unconfigured"
        elif not any(c["score"] is not None for c in alternatives):
            status = "unscored"
        elif not eligible:
            status = "low_confidence"
        elif margin is not None and (margin <= 0 or margin < limits["min_margin"]):
            status = "ambiguous"
        else:
            status = "accepted"
        selected[field] = {"value": eligible[0]["value"] if status == "accepted" else None,
                           "status": status, "margin": margin,
                           "best": alternatives[0]["observations"][0] if alternatives else None,
                           "alternatives": alternatives}
    return selected


def reselect(document, config):
    """Update selection in standalone or three-camera JSON, leaving raw evidence intact."""
    def update_view(view):
        view["fields"] = select_fields({"image": view}, config) if view["proposals"] else {}

    if "images" in document:
        for result in document["images"]:
            update_view(result)
        if document.get("same_bottle"):
            document["fields"] = select_fields({r["image_path"]: r for r in document["images"]}, config)
    elif "predictions" in document:
        for result in document["predictions"]:
            if "ocr" in result:
                for view in result["ocr"].values():
                    update_view(view)
                result["ocr_fields"] = select_fields(result["ocr"], config, result["crops"])
    else:
        raise ValueError("Expected standalone images or three-camera predictions.")
    document["ocr_selection"] = config
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved JSON containing candidates from this reader.")
    parser.add_argument("--config", type=Path, help="Selection config; defaults to ocr_selection.json.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; the input is preserved.")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new --output file.")
    try:
        document = reselect(json.loads(args.input.read_text(encoding="utf-8")), load_selection(args.config))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Invalid candidate cache or selection config: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(document, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
