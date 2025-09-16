from ultralytics import YOLO

def test_yolo_inference():
    # Load trained model
    model = YOLO("./models/best_mcd_obb.pt")

    # Run prediction
    results = model.predict("./data/images/val/image_1235.jpg", imgsz=640)

    # Access detections safely
    boxes = results[0].boxes
    assert boxes is not None, "No boxes object returned"
    assert boxes.data.shape[0] > 0, "YOLO did not detect any objects"