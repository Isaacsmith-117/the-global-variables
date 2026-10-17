# live + save:        python track_tricam.py
# live only:          python track_tricam.py --no-save
# no display + save:  python track_tricam.py --no-display



import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

DEFAULT_VIDEOS = (
    "14_55_back_left_cropped.mp4",
    "14_55_back_right_cropped.mp4",
    "14_55_front_cropped.mp4",
)
CAMERA_NAMES = ("back_left", "back_right", "front")

MATCH_MAX_AGE = 45        # frames a global track survives without being seen
MATCH_THRESHOLD = 0.62    # max Bhattacharyya distance to link two views
COLORS = [
    (0, 220, 80), (255, 160, 0), (255, 60, 200), (0, 200, 255),
    (80, 80, 255), (200, 255, 0), (255, 255, 255), (160, 100, 255),
]


class GlobalIdRegistry:
    """Map (camera, local_id) -> global id shared across all camera views."""

    def __init__(self):
        self.next_id = 1
        self.local_to_global = {}
        self.tracks = {} 
        self._frame = None

    def update(self, camera, local_id, class_id, feature, frame_no):
        if self._frame != frame_no:
            self._frame = frame_no
            for t in self.tracks.values():
                t["cams"].clear()

        key = (camera, local_id)
        gid = self.local_to_global.get(key)
        if gid is None or gid not in self.tracks:
            gid = self._find_match(class_id, feature, camera, frame_no)
        if gid is None:
            gid = self.next_id
            self.next_id += 1
            self.tracks[gid] = {
                "class_id": class_id, "feature": feature,
                "last_seen": frame_no, "cams": set(),
            }
        t = self.tracks[gid]
        t["feature"] = feature
        t["last_seen"] = frame_no
        t["cams"].add(camera)
        self.local_to_global[key] = gid
        return gid

    def _find_match(self, class_id, feature, camera, frame_no):
        best, best_d = None, MATCH_THRESHOLD
        for gid, t in self.tracks.items():
            if (t["class_id"] != class_id or camera in t["cams"]
                    or frame_no - t["last_seen"] > MATCH_MAX_AGE):
                continue
            d = cv2.compareHist(feature, t["feature"], cv2.HISTCMP_BHATTACHARYYA)
            if d < best_d:
                best, best_d = gid, d
        return best

    def cleanup(self, frame_no):
        expired = [g for g, t in self.tracks.items()
                   if frame_no - t["last_seen"] > MATCH_MAX_AGE]
        for g in expired:
            del self.tracks[g]
        if expired:
            dead = set(expired)
            self.local_to_global = {
                k: v for k, v in self.local_to_global.items() if v not in dead
            }


def appearance_feature(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 6], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten().astype(np.float32)

class VideoReader:
    def __init__(self, path, buffer=16):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open video: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.q = queue.Queue(maxsize=buffer)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.q.put(None)
                return
            while not self._stop.is_set():
                try:
                    self.q.put(frame, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def read(self):
        return self.q.get() 

    def release(self):
        self._stop.set()
        self.thread.join(timeout=2)
        self.cap.release()

def make_tracker(fps):
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    return BYTETracker(cfg)


def draw(frame, tracks, name, registry, frame_no, with_labels=True):
    """Update registry for each track; draw only when with_labels is True."""
    out = frame.copy() if with_labels else None
    for x1, y1, x2, y2, local_id, conf, cls, _ in tracks:
        gid = registry.update(
            name, int(local_id), int(cls),
            appearance_feature(frame, (x1, y1, x2, y2)), frame_no,
        )
        if out is None:
            continue
        color = COLORS[gid % len(COLORS)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, f"ID {gid} {conf:.2f}", (p1[0], max(24, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if out is not None:
        cv2.putText(out, name, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
    return out


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path,
                   default=root / "src/models/cosminCloud101Epochs.pt")
    p.add_argument("--video-dir", type=Path, default=root / "data/video")
    p.add_argument("--output", type=Path, default=root / "src/runs/tricam")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0, mps ...")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--no-display", action="store_true",
                   help="headless mode: no windows, no drawing unless saving")
    p.add_argument("--no-save", action="store_true", help="do not write videos")
    p.add_argument("--no-half", action="store_true", help="disable FP16 on CUDA")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    show = not args.no_display
    save = not args.no_save
    draw_needed = show or save

    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights missing: {weights}")
    paths = [(args.video_dir / n).resolve() for n in DEFAULT_VIDEOS]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Video missing: {path}")

    out_dir = args.output.resolve()
    if save:
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            raise SystemExit(f"{out_dir} is not empty; use --overwrite")
        out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    half = device.startswith("cuda") and not args.no_half
    model = YOLO(str(weights))
    print(f"Device: {device} | half: {half} | live: {show} | save: {save}")

    readers = [VideoReader(p) for p in paths]
    fps = min(r.fps for r in readers)
    total = min(r.total for r in readers) if all(r.total > 0 for r in readers) else 0
    trackers = [make_tracker(fps) for _ in readers]
    registry = GlobalIdRegistry()

    writers = []
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = [
            cv2.VideoWriter(str(out_dir / f"{name}.mp4"), fourcc, fps,
                            (r.width, r.height))
            for name, r in zip(CAMERA_NAMES, readers)
        ]

    dummy = [np.zeros((r.height, r.width, 3), np.uint8) for r in readers]
    model.predict(dummy, imgsz=args.imgsz, device=device, half=half, verbose=False)

    frame_no = 0
    start = time.perf_counter()
    try:
        while True:
            images = [r.read() for r in readers]
            if any(img is None for img in images):
                break

            results = model.predict(
                images, imgsz=args.imgsz, conf=args.conf, device=device,
                half=half, verbose=False,
            )

            for i, (img, res) in enumerate(zip(images, results)):
                tracks = trackers[i].update(res.boxes.cpu(), img)
                annotated = draw(img, tracks, CAMERA_NAMES[i], registry,
                                 frame_no, with_labels=draw_needed)
                if save:
                    writers[i].write(annotated)
                if show:
                    cv2.imshow(f"Camera: {CAMERA_NAMES[i]}", annotated)

            registry.cleanup(frame_no)
            frame_no += 1

            if show and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                print("Stopped by user.")
                break
            if not show and frame_no % 50 == 0:
                elapsed = time.perf_counter() - start
                progress = f"{frame_no}/{total}" if total else str(frame_no)
                print(f"\rframes {progress} | {frame_no / elapsed:.1f} sync-fps",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for r in readers:
            r.release()
        for w in writers:
            w.release()
        cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - start, 1e-6)
    print(f"\nProcessed {frame_no} synchronized frames in {elapsed:.1f}s "
          f"({frame_no / elapsed:.1f} fps x 3 cameras)")
    if save:
        print(f"Annotated videos: {out_dir}")


if __name__ == "__main__":
    main()