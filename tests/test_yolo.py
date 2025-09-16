from ultralytics import YOLO

def test_yolo_inference():
    model = YOLO("best_MCD_small.pt")  # small pre-trained model
    results = model.predict("./data/images/val/image_1235.jpg", imgsz=640)
    
    # Ensure results contain bounding boxes
    assert len(results[0].boxes) > 0, "YOLO did not detect any objects"