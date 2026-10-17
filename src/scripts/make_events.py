from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = {
    "back_left": "14_55_back_left_cropped",
    "back_right": "14_55_back_right_cropped",
    "front": "14_55_front_cropped",
}
MIN_FRAMES = 40      # shorter tracks are junk
MIN_MOVE_FRAC = 0.4  # a real bottle moves at least 40% of the image width
LINE_FRAC = 0.5      # the line sits at 50% of the width (tune per camera)


def make_events(cam, stem):
    df = pd.read_csv(ROOT / "tracks" / f"{stem}.csv")
    df["cx"] = (df.x1 + df.x2) / 2
    width = df.x2.max()  # rough image width

    stats = df.groupby("track_id").agg(
        frames=("frame", "count"),
        move=("cx", lambda s: s.max() - s.min()),
    )
    good = stats[(stats.frames >= MIN_FRAMES) & (stats.move > MIN_MOVE_FRAC * width)].index
    d = df[df.track_id.isin(good)].sort_values(["track_id", "frame"])

    # which way do bottles travel? (cx decreasing = right to left)
    first = d.groupby("track_id").first()
    last = d.groupby("track_id").last()
    right_to_left = (last.cx - first.cx).median() < 0

    # flip the sign so "progress" always increases in the travel direction
    d = d.assign(p=-d.cx if right_to_left else d.cx)
    line = width * LINE_FRAC
    line_p = -line if right_to_left else line

    rows = []
    for tid, g in d.groupby("track_id"):
        if g.p.iloc[0] >= line_p:
            continue  # already past the line at its first frame (the "zeroth" bottle)
        crossed = g[g.p >= line_p]
        if crossed.empty:
            continue  # never reached the line (cut off at the end of the video)
        rows.append((tid, int(crossed.frame.iloc[0])))

    ev = pd.DataFrame(rows, columns=["track_id", "cross_frame"]).sort_values("cross_frame")
    ev["order"] = range(1, len(ev) + 1)

    print(f"\n=== {cam} ===")
    print("direction:", "right to left" if right_to_left else "left to right")
    print("good tracks:", len(good), "| bottles that crossed the line:", len(ev))
    print("gaps between bottles (frames):")
    print(ev.cross_frame.diff().describe().round(1))
    print("suspicious gaps (<40 or >130):")
    ev["gap"] = ev.cross_frame.diff()
    print(ev[(ev.gap < 40) | (ev.gap > 130)])

    out = ROOT / "tracks" / f"events_{cam}.csv"
    ev.drop(columns="gap").to_csv(out, index=False)


for cam, stem in VIDEOS.items():
    make_events(cam, stem)