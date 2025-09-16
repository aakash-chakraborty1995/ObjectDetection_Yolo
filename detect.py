from ultralytics import YOLO

# Load a trained model (for demo use pre-trained YOLOv8n)
model = YOLO("best_MCD_small.pt")

# Run inference on a sample image
results = model("./data/images/val/image_1235.jpg")

# Print detections
for r in results:
    print(r.boxes)  # bounding boxes
