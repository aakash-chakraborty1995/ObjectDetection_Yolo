from ultralytics import YOLO

# Load trained model
model = YOLO("./models/best_mcd_obb.pt")

# Run inference
results = model("./data/images/val/image_1235.jpg")

# Print results
for r in results:
    print(r.boxes)  # bounding boxes

