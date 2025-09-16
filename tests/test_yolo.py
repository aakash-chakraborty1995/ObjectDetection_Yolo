from ultralytics import YOLO

def test_yolo_inference():
    # Load trained model
    model = YOLO("best_mcd_obb.pt")

    # Run prediction
    results = model.predict("./data/images/val/image_1235.jpg", imgsz=640)

    # Check at least one detection
    assert len(results[0].boxes) > 0, "YOLO did not detect any objects"
