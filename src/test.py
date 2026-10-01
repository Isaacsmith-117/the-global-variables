from ultralytics import YOLO

# 1. Load your newly trained custom model weights
# (Adjust path depending on where your runs folder was generated relative to src)
model = YOLO("../runs/detect/train/weights/best.pt")

# 2. Run prediction on a test image, folder, or video
source_path = ""  # Replace with your image path

results = model.predict(
    source=source_path,
    save=True,  # Automatically saves the results with drawn bounding boxes
    conf=0.5,  # Confidence threshold (only show detections above 50% certainty)
)

print("Detection complete! Check the 'runs/predict' folder for output images.")