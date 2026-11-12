# Extract the frame in which every bottle crosses the line, video by video, and write a
# reviewable dataset: crops, a CSV (one row per crop), a contact sheet per bottle and review.html.
#
# For each video:
#   1. tracks the video itself (boxes are saved to <output>/tracks and reused next time),
#      so the boxes always belong to exactly this video,
#   2. finds the real bottles and the frame in which each one crosses the line
#      (same logic as tricam_offline.py, so the numbering is the same as in output.mp4).
# Then, for the selected bottles (all of them, or a pilot subset with --pilot N):
#   3. saves the full frame, the crop and a debug image (line + box drawn on the frame),
#   4. writes manifest.csv with the columns of issue #17 (+ a few extra columns at the end),
#   5. writes contacts/b001.jpg (all views of a bottle side by side) and review.html.
#
# Crossing-order spreadsheet matches are proposals. Only --matches, a CSV of reviewed
# camera/track/frame matches, fills bottle_id and primagaz_status. Reasons stay "unknown".
#
# Needs: pip install openpyxl   (only to read the .xlsx spreadsheet)
#
# Pass --recording-id, --videos, --truth and a new --output folder explicitly.
# See README.md for a command and the reviewed-match CSV format.

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__:
    from .config import PROJECT_ROOT
    from .tricam_offline import LINE_FRAC, crossings, track_to_csv
else:
    from config import PROJECT_ROOT
    from tricam_offline import LINE_FRAC, crossings, track_to_csv

# ---- default paths, all relative to the project root (same videos as tricam_offline.py).
# If your files are somewhere else, use the command line flags instead of editing this file.
DEFAULT_WEIGHTS = PROJECT_ROOT / "src" / "models" / "cosminCloud101Epochs.pt"

EDGE_MARGIN = 4      # px: a box this close to the image border counts as clipped
SYNC_TOLERANCE = 15  # frames: a bottle whose crossing delay between cameras is further off is flagged

# The columns of issue #17, then extra columns that help the review.
COLUMNS = [
    "image_path", "recording_id", "camera_id", "frame_index", "bottle_id", "track_id",
    "source_row", "primagaz_status", "match_verified", "major_damage", "heavy_dirt",
    "wrong_brand_or_model", "review_status", "notes",
    "proposed_bottle_id", "spreadsheet_id", "spreadsheet_color", "clipped", "sync_dev_frames",
    "conf", "frame_path", "contact_sheet_path",
    "candidate_source_row", "candidate_spreadsheet_id", "candidate_primagaz_status",
    "candidate_spreadsheet_color",
]


def camera_name(video, recording_id):
    name = video.stem.removeprefix(recording_id + "_").removesuffix("_cropped")
    if not video.stem.startswith(recording_id + "_") or name not in (
            "front", "back_left", "back_right", "top"):
        raise SystemExit(f"Video name does not identify a {recording_id} camera: {video.name}")
    return name


def cell(info, key):
    """A value of a spreadsheet row, or "" if there is no row or the cell is empty."""
    if not info or pd.isna(info.get(key)):
        return ""
    return info[key]


def load_truth(path):
    """Read the ground-truth spreadsheet (xlsx/csv/tsv) into {bottle_id: row dict}."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")  # sniffs tab / comma / semicolon
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ("id", "primagaz_status"):
        if col not in df.columns:
            raise SystemExit(f"Spreadsheet needs an '{col}' column, found: {list(df.columns)}")

    df["source_row"] = df.index + 2  # row in the sheet (row 1 is the header)
    ids = pd.to_numeric(df["id"], errors="coerce")
    if (df["id"].notna() & ids.isna()).any() or not np.isfinite(ids.dropna()).all():
        raise SystemExit("Spreadsheet IDs must be finite integers.")
    df["id"] = ids
    dropped = int(df["id"].isna().sum())
    if dropped:
        print(f"Skipping {dropped} spreadsheet rows without a valid id.")
    df = df.dropna(subset=["id"])
    if df.empty or (df["id"] % 1 != 0).any() or df["id"].duplicated().any():
        raise SystemExit("Spreadsheet IDs must be unique integers.")

    # Preserve the plant value; only reviewed matches use its label as a folder name.
    status = df["primagaz_status"]
    df["status_raw"] = status.where(status.notna(), "").astype(str)
    label = df["status_raw"].str.strip().str.lower()
    df["status_label"] = label.where(label != "", "unknown")
    return {int(r["id"]): r for r in df.to_dict("records")}


def load_matches(path, recording_id, cameras, truth):
    """Each CSV row is a manually reviewed link for one camera crossing."""
    if path is None:
        return {}
    columns = ["recording_id", "camera_id", "track_id", "frame_index", "spreadsheet_id"]
    df = pd.read_csv(path, dtype=str).fillna("")
    if not set(columns) <= set(df.columns):
        raise SystemExit(f"Matches CSV needs columns: {', '.join(columns)}")
    matches, assigned = {}, set()
    for row in df[columns].itertuples(index=False):
        if row.recording_id != recording_id or row.camera_id not in cameras:
            raise SystemExit("Matches CSV contains another recording or an unselected camera.")
        try:
            track, frame, spreadsheet_id = map(int, row[2:])
        except ValueError as error:
            raise SystemExit("Match track, frame and spreadsheet IDs must be integers.") from error
        key = (row.camera_id, track, frame)
        if key in matches or (row.camera_id, spreadsheet_id) in assigned:
            raise SystemExit("Matches CSV repeats a crossing or a spreadsheet ID in one camera.")
        if spreadsheet_id not in truth or truth[spreadsheet_id]["status_label"] not in ("ok", "nok"):
            raise SystemExit(f"Match has no valid plant reference: {spreadsheet_id}")
        matches[key] = spreadsheet_id
        assigned.add((row.camera_id, spreadsheet_id))
    return matches


def file_info(path):
    """Record the exact local input used by this extraction."""
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def save_image(path, image, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.size == 0 or not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"Could not save image: {path}")


def sync_deviations(events):
    """{(bottle, camera index): frames away from the typical delay to camera 0}.
    The same bottle should cross the line with the same delay between two cameras, so a
    bottle far off that delay hints at a counting/matching error."""
    common = min(len(e) for e in events)
    dev = {}
    for i, e in enumerate(events):
        delays = [e[n][1] - events[0][n][1] for n in range(common)]
        if not delays:
            continue
        typical = float(np.median(delays))
        for n, d in enumerate(delays, start=1):
            dev[(n, i)] = int(round(abs(d - typical)))
    return dev


def spread(items, k):
    """k items spread evenly over the list."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    if k == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def select_bottles(count, truth, id_offset, pilot):
    """Bottle numbers to export: all of them, or `pilot` bottles, half NOK and half OK,
    spread over the whole recording."""
    bottles = list(range(1, count + 1))
    if not pilot:
        return bottles

    def status(n):
        row = truth.get(n + id_offset)
        return row["status_label"] if row else None

    nok = [n for n in bottles if status(n) == "nok"]
    ok = [n for n in bottles if status(n) == "ok"]
    k_nok = min(len(nok), pilot // 2)
    k_ok = min(len(ok), pilot - k_nok)
    k_nok = min(len(nok), pilot - k_ok)  # top up if there are not enough OK bottles
    selected = spread(nok, k_nok) + spread(ok, k_ok)
    remaining = [n for n in bottles if n not in selected]
    return sorted(selected + spread(remaining, min(pilot, count) - len(selected)))


def read_frames(video, wanted):
    """Read only the wanted frame numbers, sequentially (seeking in mp4 can be inexact)."""
    found = {}
    if not wanted:
        return found
    cap = cv2.VideoCapture(str(video))
    for f in range(max(wanted) + 1):
        if not cap.grab():
            break
        if f in wanted:
            ok, img = cap.retrieve()
            if ok:
                found[f] = img
    cap.release()
    return found


def crop_box(img, r, pad):
    h, w = img.shape[:2]
    bw, bh = r.x2 - r.x1, r.y2 - r.y1
    x1, x2 = max(0, int(r.x1 - pad * bw)), min(w, int(r.x2 + pad * bw))
    y1, y2 = max(0, int(r.y1 - pad * bh)), min(h, int(r.y2 + pad * bh))
    return img[y1:y2, x1:x2]


def is_clipped(r, w, h):
    """True if the box touches the image border, so part of the bottle may be missing."""
    return bool(r.x1 <= EDGE_MARGIN or r.y1 <= EDGE_MARGIN
                or r.x2 >= w - EDGE_MARGIN or r.y2 >= h - EDGE_MARGIN)


def debug_image(img, r, bottle):
    """The frame with the counting line, the box and the number, shrunk to 960 px wide."""
    img = img.copy()
    h, w = img.shape[:2]
    s = w / 960
    thick = max(2, round(3 * s))
    cv2.line(img, (int(LINE_FRAC * w), 0), (int(LINE_FRAC * w), h), (0, 255, 255), thick)
    cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 220, 80), thick)
    cv2.putText(img, f"ID {bottle}", (int(r.x1), max(int(40 * s), int(r.y1) - int(10 * s))),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3 * s, (0, 220, 80), thick)
    return cv2.resize(img, (960, round(h * 960 / w)))


def make_contact_sheet(rows, title, output_dir):
    """All views of one bottle side by side, with a caption bar. Returns the image or None."""
    tiles = []
    for r in sorted(rows, key=lambda r: r["camera_id"]):
        img = cv2.imread(str(output_dir / r["image_path"]))
        if img is None:
            continue
        height = 360
        img = cv2.resize(img, (max(1, round(img.shape[1] * height / img.shape[0])), height))
        cv2.putText(img, r["camera_id"], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, r["camera_id"], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(cv2.copyMakeBorder(img, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    if not tiles:
        return None
    strip = np.hstack(tiles)
    bar = np.full((44, max(strip.shape[1], 600), 3), 40, np.uint8)
    cv2.putText(bar, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    sheet = np.full((bar.shape[0] + strip.shape[0], bar.shape[1], 3), 40, np.uint8)
    sheet[:bar.shape[0]] = bar
    sheet[bar.shape[0]:, :strip.shape[1]] = strip
    return sheet


def write_review_html(path, cards):
    """One page with a card per bottle: its crops (click = full frame) and the data to check."""
    n_status = {s: sum(c["status"] == s for c in cards) for s in ("ok", "nok")}
    n_flagged = sum(bool(c["warnings"]) for c in cards)
    out = [f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Dataset review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#1b1b1b; --card:#f3f3f3; --warn:#b3261e; --line:#d0d0d0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181a; --fg:#e8e8e8; --card:#23262a; --warn:#ff8a80; --line:#3a3e44; }} }}
body {{ font-family: system-ui, sans-serif; background:var(--bg); color:var(--fg); margin:0; padding:1rem 1.5rem; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:.8rem 1rem; margin:0 0 1rem; }}
.card h2 {{ margin:.1rem 0 .3rem; font-size:1.1rem; }}
.meta {{ font-size:.9rem; margin-bottom:.5rem; }}
.warn {{ color:var(--warn); font-weight:600; }}
.views {{ display:flex; flex-wrap:wrap; gap:.6rem; }}
figure {{ margin:0; }}
img {{ height:260px; max-width:100%; object-fit:contain; border:1px solid var(--line); background:#000; }}
figcaption {{ font-size:.8rem; }}
</style></head><body>
<h1>Dataset review</h1>
<p>{len(cards)} groups ({n_status['ok']} verified OK, {n_status['nok']} verified NOK), {n_flagged} with warnings.
Crossing-order groups and candidate spreadsheet references require review.
Verified groups use the supplied reviewed matches. Reason labels remain pending.
Click an image to open the full frame.</p>
"""]
    for c in cards:
        warn = "".join(f' <span class="warn">⚠ {html.escape(w)}</span>' for w in c["warnings"])
        out.append(f'<div class="card"><h2>{html.escape(c["title"])}</h2>'
                   f'<div class="meta">{html.escape(c["meta"])}{warn}</div><div class="views">')
        for r in c["rows"]:
            note = f'{r["camera_id"]} · frame {r["frame_index"]}' + (" · CLIPPED" if r["clipped"] else "")
            out.append(f'<figure><a href="{html.escape(r["frame_path"])}">'
                       f'<img loading="lazy" src="{html.escape(r["image_path"])}" alt="{html.escape(note)}"></a>'
                       f'<figcaption>{html.escape(note)}</figcaption></figure>')
        out.append("</div></div>")
    out.append("</body></html>")
    path.write_text("\n".join(out), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description="Extract the crossing frame of every bottle, per video.")
    p.add_argument("--videos", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--tracks-dir", type=Path, default=None,
                   help="folder with one <video name>.csv per video (default: <output>/tracks)")
    p.add_argument("--retrack", action="store_true", help="track the videos again")
    p.add_argument("--truth", type=Path, required=True, help="recording's ground-truth spreadsheet")
    p.add_argument("--recording-id", required=True)
    p.add_argument("--matches", type=Path, help="CSV of manually reviewed camera/track/frame matches")
    p.add_argument("--id-offset", type=int, default=0,
                   help="candidate spreadsheet id = crossing number + this")
    p.add_argument("--pilot", type=int, default=0,
                   help="export only N bottles, half OK and half NOK (0 = all bottles)")
    p.add_argument("--pad", type=float, default=0.08, help="crop padding as a fraction of the box")
    p.add_argument("--output", type=Path, required=True, help="new output folder")
    args = p.parse_args(argv)

    if args.output.exists():
        raise SystemExit("Output already exists; choose a new --output folder.")
    if args.pilot < 0 or not np.isfinite(args.pad) or args.pad < 0:
        raise SystemExit("Pilot count and padding must be nonnegative.")
    if args.tracks_dir and args.retrack:
        raise SystemExit("Use either --tracks-dir or --retrack.")
    cams = [camera_name(video, args.recording_id) for video in args.videos]
    if len(cams) != len(set(cams)):
        raise SystemExit("Select each camera only once.")
    if not args.truth.stem.startswith(args.recording_id + "_ground_truth"):
        raise SystemExit("Workbook name does not match --recording-id.")
    for path in [args.truth, *args.videos, *([args.matches] if args.matches else [])]:
        if not path.is_file():
            raise SystemExit(f"File missing: {path}")
    truth = load_truth(args.truth)
    matches = load_matches(args.matches, args.recording_id, cams, truth)
    run = {"complete": False, "recording_id": args.recording_id, "pad": args.pad,
           "pilot": args.pilot, "id_offset": args.id_offset,
           "extractor": file_info(Path(__file__)),
           "truth": file_info(args.truth), "matches": file_info(args.matches) if args.matches else None,
           "cameras": []}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    tracks_dir = args.tracks_dir or args.output / "tracks"
    if not args.tracks_dir:
        tracks_dir.mkdir()

    # ---- phase 1: tracks and crossing frames of every video
    tracks, events, sizes = [], [], []
    for video, cam in zip(args.videos, cams):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"OpenCV could not open {video} (codec? re-encode to H.264 with ffmpeg).")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        csv_path = tracks_dir / f"{video.stem}.csv"
        if args.tracks_dir and not csv_path.is_file():
            raise SystemExit(f"Saved tracks missing: {csv_path}")
        reused = csv_path.is_file() and not args.retrack
        if reused:
            print(f"[{cam}] using saved tracks {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            if not args.weights.is_file():
                raise SystemExit(f"Weights missing: {args.weights}")
            print(f"[{cam}] tracking {video.name} (slow, only the first time) ...")
            df = track_to_csv(args.weights, video, csv_path)
        columns = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]
        if not set(columns) <= set(df.columns) or df.empty:
            raise SystemExit(f"Tracking CSV is empty or missing columns: {csv_path}")
        df = df[columns].apply(pd.to_numeric, errors="raise")
        if (not np.isfinite(df.to_numpy()).all() or df.duplicated(["track_id", "frame"]).any()
                or (df[["frame", "track_id"]] % 1 != 0).any().any()
                or df.frame.min() < 0 or df.frame.max() >= total or df.track_id.min() < 0
                or df.x1.min() < -5 or df.y1.min() < -5 or df.x2.max() > w + 5 or df.y2.max() > h + 5
                or (df.x2 <= df.x1).any() or (df.y2 <= df.y1).any()
                or not df.conf.between(0, 1).all()):
            raise SystemExit(f"Tracking CSV does not fit this video: {csv_path}")
        e = crossings(df, w)
        print(f"[{cam}] {len(e)} bottles")
        tracks.append(df)
        events.append(e)
        sizes.append((w, h))
        run["cameras"].append({"camera_id": cam, "video": file_info(video),
                               "tracks": file_info(csv_path), "crossings": len(e),
                               "width": w, "height": h, "frames": total,
                               "reused_tracks": reused,
                               "weights": None if reused else file_info(args.weights)})
        (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    crossing_keys = {(cam, int(track), frame) for cam, camera_events in zip(cams, events)
                     for track, frame in camera_events}
    if set(matches) - crossing_keys:
        raise SystemExit("Matches CSV refers to a track/frame that is not a crossing in these videos.")

    if min(len(e) for e in events) == 0:
        raise SystemExit("A camera found 0 bottles, nothing to export.")
    common = min(len(e) for e in events)
    if len({len(e) for e in events}) > 1:
        print(f"WARNING: the cameras counted different numbers of bottles; numbers after a missed "
              f"bottle can be shifted in some cameras. Using the first {common} for the sync check.")
    dev = sync_deviations(events)

    # ---- which bottles to export
    selected = None
    if args.pilot:
        selected = set(select_bottles(common, truth, args.id_offset, args.pilot))
        print(f"Pilot: bottles {sorted(selected)}")

    # ---- phase 2: save frames and crops, collect the CSV rows
    rows = []
    for i, video in enumerate(args.videos):
        cam, df, (w, h) = cams[i], tracks[i], sizes[i]
        wanted = {f for n, (_, f) in enumerate(events[i], start=1) if selected is None or n in selected}
        images = read_frames(video, wanted)

        for bottle, (tid, f) in enumerate(events[i], start=1):
            if selected is not None and bottle not in selected:
                continue
            if f not in images:
                raise RuntimeError(f"Could not read {cam} frame {f}.")
            r = next(df[(df.track_id == tid) & (df.frame == f)].itertuples())
            candidate = truth.get(bottle + args.id_offset)
            spreadsheet_id = matches.get((cam, int(tid), f))
            info = truth.get(spreadsheet_id)
            label = info["status_label"] if info else "unknown"
            name = f"b{bottle:03d}_{cam}.jpg"
            frame_rel, crop_rel = f"frames/{label}/{name}", f"crops/{label}/{name}"
            for rel, img in ((frame_rel, images[f]), (crop_rel, crop_box(images[f], r, args.pad))):
                save_image(args.output / rel, img)
            dbg = args.output / "debug" / cam / f"b{bottle:03d}.jpg"
            save_image(dbg, debug_image(images[f], r, bottle))

            rows.append({
                "_group": (int(info is not None), spreadsheet_id if info else bottle),
                "image_path": crop_rel, "recording_id": args.recording_id, "camera_id": cam,
                "frame_index": f, "bottle_id": spreadsheet_id if info else "", "track_id": int(tid),
                "source_row": f"{args.truth.name}:{info['source_row']}" if info else "",
                "primagaz_status": cell(info, "status_raw"),
                "match_verified": int(info is not None),
                "major_damage": "unknown", "heavy_dirt": "unknown", "wrong_brand_or_model": "unknown",
                "review_status": "pending", "notes": "",
                "proposed_bottle_id": f"{args.recording_id}_b{bottle:03d}",
                "spreadsheet_id": int(info["id"]) if info else "",
                "spreadsheet_color": cell(info, "color"),
                "clipped": int(is_clipped(r, w, h)),
                "sync_dev_frames": dev.get((bottle, i), ""),
                "conf": r.conf, "frame_path": frame_rel, "contact_sheet_path": "",
                "candidate_source_row": f"{args.truth.name}:{candidate['source_row']}" if candidate else "",
                "candidate_spreadsheet_id": int(candidate["id"]) if candidate else "",
                "candidate_primagaz_status": cell(candidate, "status_raw"),
                "candidate_spreadsheet_color": cell(candidate, "color"),
            })

    # ---- phase 3: contact sheets, review page, CSV
    rows.sort(key=lambda r: (r["_group"], r["camera_id"]))
    cards = []
    for n, key in enumerate(sorted({r["_group"] for r in rows}), start=1):
        group = [r for r in rows if r["_group"] == key]
        first = group[0]
        status = first["primagaz_status"].strip().lower() or "unknown"
        devs = [r["sync_dev_frames"] for r in group if r["sync_dev_frames"] != ""]
        worst = max(devs) if devs else None
        warnings = []
        if not first["match_verified"]:
            warnings.append("camera grouping and spreadsheet references are proposed; review required")
        if len(group) != len(cams):
            warnings.append("this group is missing selected camera views")
        if any(r["clipped"] for r in group):
            warnings.append("box touches the image border in at least one view")
        if worst is not None and worst > SYNC_TOLERANCE:
            warnings.append(f"cameras disagree on the crossing time by {worst} frames, check the match")

        meta = (f"status {first['primagaz_status'] or '?'} · colour {first['spreadsheet_color'] or '?'} · "
                f"spreadsheet id {first['spreadsheet_id']} · {first['source_row'] or 'no row'}"
                f" · candidate id {first['candidate_spreadsheet_id']} · "
                f"candidate status {first['candidate_primagaz_status'] or '?'}"
                + (f" · sync deviation {worst} frames" if worst is not None else ""))
        title = (f"{args.recording_id} bottle {first['bottle_id']} [VERIFIED {status.upper()}]"
                 if first["match_verified"] else f"{first['proposed_bottle_id']} [PROPOSED]")
        sheet = make_contact_sheet(group, f"{title} | colour {first['spreadsheet_color'] or '?'} | "
                                          f"sync dev {worst if worst is not None else '?'}", args.output)
        if sheet is not None:
            rel = f"contacts/b{n:03d}.jpg"
            (args.output / "contacts").mkdir(parents=True, exist_ok=True)
            save_image(args.output / rel, sheet, quality=90)
            for r in group:
                r["contact_sheet_path"] = rel
        cards.append({"title": title, "meta": meta, "status": status, "warnings": warnings, "rows": group})

    pd.DataFrame(rows, columns=COLUMNS).to_csv(args.output / "manifest.csv", index=False)
    write_review_html(args.output / "review.html", cards)
    run.update(complete=True, crops=len(rows), verified_crops=sum(r["match_verified"] for r in rows))
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    flagged = sum(bool(c["warnings"]) for c in cards)
    print(f"Saved {len(rows)} crops of {len(cards)} bottles to {args.output}")
    print(f"{flagged} bottles have warnings. Open {args.output / 'review.html'} to check them.")
    print("Remember: keep this folder out of Git (images and decisions are private).")


if __name__ == "__main__":
    main()
