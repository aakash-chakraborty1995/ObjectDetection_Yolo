from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11s-obb.yaml")  # build a new model from YAML
# model = YOLO("yolo11s-obb.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11s-obb.yaml").load("yolo11n-obb.pt")  # build from YAML and transfer weights

# Load a model
model = YOLO("yolo11s-obb.yaml")  # build a new model from YAML
model = YOLO("yolo11s-obb.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11s-obb.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="multicrate.yaml", epochs=100, imgsz=640, task = 'obb')

model.save("yolo11s-obb.pt")
model.export(format="onnx")