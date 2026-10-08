from ultralytics import YOLO

model = YOLO("../runs/detect/train/weights/best.pt")

model.track(
    source="../data/video/filling_plant_bolivia.mp4",
    show=True,
    conf=0.5,
)