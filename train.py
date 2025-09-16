from ultralytics import YOLO

model = YOLO("yolo11s-obb.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="multicrate.yaml", epochs=100, imgsz=640, task = 'obb')

model.save("yolo11s-mcd-obb.pt")
model.export(format="onnx")