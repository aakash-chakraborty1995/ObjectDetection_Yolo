import sagemaker
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::016873651001:role/SageMaker-YOLO-Role"  # attach AmazonSageMakerFullAccess + S3 permissions
sess = sagemaker.Session()

# Define S3 locations
input_s3 = "s3://mcdyolobucket/"
output_s3 = "s3://mcdyolobucket/yolo-output/"

# Estimator (uses SageMaker built-in PyTorch container)
estimator = PyTorch(
    entry_point="train_sm.py",
    source_dir=".",
    role=role,
    framework_version="1.13",
    py_version="py39",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",  # GPU
    output_path=output_s3,
    hyperparameters={
        "epochs": 100,
        "imgsz": 640
    }
)

# Launch training
estimator.fit({"training": input_s3})
