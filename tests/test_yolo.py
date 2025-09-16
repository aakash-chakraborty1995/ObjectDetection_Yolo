from ultralytics import YOLO

def test_yolo_inference():
    # Load trained model
    model = YOLO("./models/best_mcd_obb.pt")

    # Run prediction
    results = model.predict("./data/images/val/image_1235.jpg", imgsz=640)

    # Access the results
    for result in results:
        xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
        xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
        names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
        confs = result.obb.conf  # confidence score of each box