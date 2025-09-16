from ultralytics import YOLO

model = YOLO("yolo11m-obb.yaml").load("yolo11m.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="multicrate.yaml", epochs=100, imgsz=640, task = 'obb')

model.save("./models/best_mcd_obb.pt")
model.export(format="onnx")