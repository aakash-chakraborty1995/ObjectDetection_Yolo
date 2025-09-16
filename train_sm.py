import os
from ultralytics import YOLO

# SageMaker provides a model output directory
output_dir = os.environ.get("SM_MODEL_DIR", "./models")

# Load pretrained model
model = YOLO("yolo11m-obb.yaml").load("yolo11m.pt")

# Train with dataset YAML stored in S3
results = model.train(
    data="s3://mcdyolobucket/multicrate_sm.yml", 
    epochs=100, 
    imgsz=640, 
    task="obb"
)

# Save the best weights inside the SageMaker model dir
best_model_path = os.path.join(output_dir, "best_mcd_obb.pt")
model.save(best_model_path)

# Also export ONNX inside model dir
model.export(format="onnx", imgsz=640, dynamic=True, optimize=True, half=True, device=0)
