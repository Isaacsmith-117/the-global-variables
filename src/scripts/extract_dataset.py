# Extract the frame in which every bottle crosses the line, video by video.
#
# Each video is handled on its own (no synchronization between cameras needed):
#   1. tracks the video itself (boxes are saved to <output>/tracks and reused next time),
#      so the boxes always belong to exactly this video,
#   2. finds the real bottles and the frame in which each one crosses the line
#      (same logic as tricam_offline.py, so the global ID n is the same as in output.mp4),
#   3. saves for every bottle: the full frame, a crop, and a debug image with the line,
#      the box and the ID drawn on it, named b<ID>_<camera>_<label>.jpg,
#   4. writes manifest.csv.
#
# Needs: pip install openpyxl   (only to read the .xlsx spreadsheet)
#
# Expected layout (relative to the project root):
#   data/video/14_55_back_left_cropped.mp4, ..._back_right_..., ..._front_...
#   data/14_55_ground_truth_ai_app.xlsx
#   src/models/cosminCloud101Epochs.pt
# Files somewhere else? Pass --videos, --truth, --weights.
#
# python src/scripts/extract_dataset.py
# python src/scripts/extract_dataset.py --videos a.mp4 b.mp4 c.mp4 --truth gt.xlsx
# python src/scripts/extract_dataset.py --tracks-dir runs/tricam_offline   (use existing CSVs)
# python src/scripts/extract_dataset.py --retrack                          (track again)

import argparse
from pathlib import Path

import cv2
import pandas as pd

if __package__:
    from .config import PROJECT_ROOT
    from .tricam_offline import DEFAULT_VIDEOS, LINE_FRAC, crossings, track_to_csv
else:
    from config import PROJECT_ROOT
    from tricam_offline import DEFAULT_VIDEOS, LINE_FRAC, crossings, track_to_csv

# ---- default paths, all relative to the project root (same videos as tricam_offline.py).
# If your files are somewhere else, use the command line flags instead of editing this file.
DEFAULT_WEIGHTS = PROJECT_ROOT / "src" / "models" / "cosminCloud101Epochs.pt"
DEFAULT_TRUTH = PROJECT_ROOT / "data" / "14_55_ground_truth_ai_app.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "dataset"


def camera_name(video):
    return video.stem.replace("14_55_", "").replace("_cropped", "")


def load_truth(path, label_column):
    """Read the ground-truth spreadsheet (xlsx/csv/tsv) into {bottle_id: row dict}."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")  # sniffs tab / comma / semicolon
    df.columns = [str(c).strip().lower() for c in df.columns]
    label_column = label_column.lower()
    if label_column not in df.columns or "id" not in df.columns:
        raise SystemExit(f"Spreadsheet needs 'id' and '{label_column}' columns, found: {list(df.columns)}")

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    dropped = int(df["id"].isna().sum())
    if dropped:
        print(f"Skipping {dropped} spreadsheet rows without a valid id.")
    df = df.dropna(subset=["id"])

    label = df[label_column].astype(str).str.strip().str.lower()
    df["label"] = label.where(~label.isin(["", "nan", "none"]), "unknown")
    return {int(r["id"]): r for r in df.to_dict("records")}


def read_frames(video, wanted):
    """Read only the wanted frame numbers, sequentially (seeking in mp4 can be inexact)."""
    cap = cv2.VideoCapture(str(video))
    found = {}
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


def debug_image(img, r, bottle):
    """The frame with the counting line, the box and the ID, shrunk to 960 px wide."""
    img = img.copy()
    h, w = img.shape[:2]
    s = w / 960
    thick = max(2, round(3 * s))
    cv2.line(img, (int(LINE_FRAC * w), 0), (int(LINE_FRAC * w), h), (0, 255, 255), thick)
    cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 220, 80), thick)
    cv2.putText(img, f"ID {bottle}", (int(r.x1), max(int(40 * s), int(r.y1) - int(10 * s))),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3 * s, (0, 220, 80), thick)
    return cv2.resize(img, (960, round(h * 960 / w)))


def main(argv=None):
    p = argparse.ArgumentParser(description="Extract the crossing frame of every bottle, per video.")
    p.add_argument("--videos", type=Path, nargs="+", default=DEFAULT_VIDEOS)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--tracks-dir", type=Path, default=None,
                   help="folder with one <video name>.csv per video (default: <output>/tracks)")
    p.add_argument("--retrack", action="store_true", help="track the videos again")
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH, help="ground-truth spreadsheet")
    p.add_argument("--label-column", default="Primagaz_status",
                   help="column used as the label (Primagaz_status or classification)")
    p.add_argument("--id-offset", type=int, default=0,
                   help="spreadsheet id = bottle number + this")
    p.add_argument("--pad", type=float, default=0.08, help="crop padding as a fraction of the box")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    for path in [args.truth, *args.videos]:
        if not path.is_file():
            raise SystemExit(f"File missing: {path}\n"
                             f"Put it at that path or pass --videos / --truth (see --help).")
    truth = load_truth(args.truth, args.label_column)
    tracks_dir = args.tracks_dir or args.output / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for video in args.videos:
        cam = camera_name(video)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"OpenCV could not open {video} (codec? re-encode to H.264 with ffmpeg).")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ---- tracks of this video
        csv_path = tracks_dir / f"{video.stem}.csv"
        if csv_path.is_file() and not args.retrack:
            print(f"[{cam}] using saved tracks {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            if not args.weights.is_file():
                raise SystemExit(f"Weights missing: {args.weights}")
            print(f"[{cam}] tracking {video.name} (slow, only the first time) ...")
            df = track_to_csv(args.weights, video, csv_path)
        if df.frame.max() >= total or df.x2.max() > w + 5:
            print(f"[{cam}] WARNING: the CSV does not fit this video "
                  f"(last frame {df.frame.max()} vs {total} frames, max x2 {df.x2.max():.0f} vs width {w}). "
                  f"Use --retrack.")

        # ---- real bottles and their crossing frames
        events = crossings(df, w)
        print(f"[{cam}] {len(events)} bottles")
        if not events:
            continue
        images = read_frames(video, {f for _, f in events})

        for bottle, (tid, f) in enumerate(events, start=1):
            if f not in images:
                print(f"[{cam}] could not read frame {f}")
                continue
            r = next(df[(df.track_id == tid) & (df.frame == f)].itertuples())
            info = truth.get(bottle + args.id_offset)
            label = info["label"] if info else "unknown"
            name = f"b{bottle:03d}_{cam}_{label}.jpg"
            outputs = (("frames", images[f]), ("crops", crop_box(images[f], r, args.pad)))
            for kind, img in outputs:
                out = args.output / kind / label / name
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            dbg = args.output / "debug" / cam / f"b{bottle:03d}.jpg"
            dbg.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dbg), debug_image(images[f], r, bottle))

            manifest.append({
                "bottle": bottle, "camera": cam, "track_id": tid, "frame": f, "conf": r.conf,
                "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2, "label": label,
                "color": info.get("color") if info else None,
                "year": info.get("year") if info else None,
                "tare_kg": info.get("pr_tare_kg") if info else None,
                "frame_path": f"frames/{label}/{name}", "crop_path": f"crops/{label}/{name}",
            })
        missing = [n for n in range(1, len(events) + 1) if (n + args.id_offset) not in truth]
        if missing:
            print(f"[{cam}] no spreadsheet row for bottles {missing} (saved under 'unknown')")

    if manifest:
        pd.DataFrame(manifest).sort_values(["bottle", "camera"]).to_csv(
            args.output / "manifest.csv", index=False)
    print(f"Saved {len(manifest)} images to {args.output}")
    print("Check a few images in the 'debug' folder: the box should sit on the yellow line.")


if __name__ == "__main__":
    main()