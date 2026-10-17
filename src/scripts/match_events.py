from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAMS = ["back_left", "back_right", "front"]

ev = {c: pd.read_csv(ROOT / "tracks" / f"events_{c}.csv").set_index("order") for c in CAMS}

# global_id = position in the sequence
m = pd.DataFrame({c: ev[c].cross_frame for c in CAMS})
m["off_right"] = m.back_right - m.back_left
m["off_front"] = m.front - m.back_left
print(m[["off_right", "off_front"]].describe().round(1))

# bottles whose offset differs a lot from the typical one
for col in ["off_right", "off_front"]:
    bad = m[(m[col] - m[col].median()).abs() > 15]
    print(f"\nOutliers in {col}:")
    print(bad)

# mapping table: (camera, track_id) -> global_id
rows = []
for c in CAMS:
    for order, r in ev[c].iterrows():
        rows.append((c, int(r.track_id), order))
mapping = pd.DataFrame(rows, columns=["camera", "track_id", "global_id"])
mapping.to_csv(ROOT / "tracks" / "mapping.csv", index=False)
print("\nSaved mapping.csv with", len(mapping), "rows")