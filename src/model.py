from ultralytics import YOLO

# Load a pre-trained nano model for maximum speed
model = YOLO("yolov8n.pt")  # or 'yolo11n.pt'

# Train the model on your custom gas bottle dataset
model.train(
    data="../data/data.yaml",
    epochs=3,  # Just 3 epochs to see if it runs
    imgsz=416,  # Lower resolution speeds up the test
    batch=8,  # Small batch size to prevent memory errors
    fraction=0.1,  # Uses only 10% of your training dataset
)