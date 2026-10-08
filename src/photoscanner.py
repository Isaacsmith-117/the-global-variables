from pathlib import Path

from ultralytics import YOLO

model = YOLO("../runs/detect/train/weights/best.pt")
input_dir = Path("../data/scannertest")

image_paths = sorted(
    path
    for path in input_dir.iterdir()
    if path.is_file()
    and path.suffix.lower() == ".jpg"
    and not path.stem.lower().endswith("checked")
)

for image_path in image_paths:
    output_path = image_path.with_name(f"{image_path.stem}checked.png")
    results = model.predict(source=str(image_path), conf=0.5, verbose=False)

    detections = 0
    for result in results:
        result.save(filename=str(output_path))
        detections += len(result.boxes)

    print(f"{image_path.name}: {detections} detection(s) -> {output_path.name}")