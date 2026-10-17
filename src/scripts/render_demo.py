from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT.parent / "videos" / "videos" / "14_55"   # edit if yours differs
CAMS = {
    "back_left": "14_55_back_left_cropped",
    "back_right": "14_55_back_right_cropped",
    "front": "14_55_front_cropped",
}
OUT = ROOT / "tracks" / "demo.mp4"
HEIGHT = 480          # every view is resized to this height
MAX_FRAMES = 700      # ~30 s for a quick test; set to None for the full video

# 1. offsets: how many frames later each camera sees the same bottle than back_left
events = {c: pd.read_csv(ROOT / "tracks" / f"events_{c}.csv").set_index("order") for c in CAMS}
offset = {c: int(round((events[c].cross_frame - events["back_left"].cross_frame).median())) for c in CAMS}
print("offsets (frames):", offset)

# 2. boxes per camera and frame, with the global id attached
mapping = pd.read_csv(ROOT / "tracks" / "mapping.csv")
boxes = {}
for c, stem in CAMS.items():
    df = pd.read_csv(ROOT / "tracks" / f"{stem}.csv")
    m = mapping[mapping.camera == c][["track_id", "global_id"]]
    df = df.merge(m, on="track_id")          # keeps only the real bottles
    boxes[c] = {f: g for f, g in df.groupby("frame")}

# 3. open the videos and jump to the common start
caps = {c: cv2.VideoCapture(str(VIDEO_DIR / f"{stem}.mp4")) for c, stem in CAMS.items()}
total = {c: int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT)) for c in CAMS}
t_start = max(0, max(-o for o in offset.values()))          # camera frame = t + offset >= 0
t_end = min(total[c] - offset[c] for c in CAMS)
if MAX_FRAMES:
    t_end = min(t_end, t_start + MAX_FRAMES)
for c in CAMS:
    caps[c].set(cv2.CAP_PROP_POS_FRAMES, t_start + offset[c])

writer = None
for t in range(t_start, t_end):
    tiles = []
    for c in CAMS:
        ok, frame = caps[c].read()
        if not ok:
            frame = np.zeros((HEIGHT, HEIGHT, 3), np.uint8)
        f_idx = t + offset[c]
        if f_idx in boxes[c]:
            for r in boxes[c][f_idx].itertuples():
                cv2.rectangle(frame, (r.x1, r.y1), (r.x2, r.y2), (0, 220, 80), 3)
                cv2.putText(frame, f"ID {r.global_id}", (r.x1, max(40, r.y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 80), 4)
        cv2.putText(frame, c, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        scale = HEIGHT / frame.shape[0]
        tiles.append(cv2.resize(frame, (int(frame.shape[1] * scale), HEIGHT)))
    row = np.hstack(tiles)
    if writer is None:
        writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 23,
                                 (row.shape[1], row.shape[0]))
    writer.write(row)

writer.release()
for cap in caps.values():
    cap.release()
print("Saved", OUT)