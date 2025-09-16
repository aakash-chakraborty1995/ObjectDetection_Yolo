# 🐶 YOLO Object Detection Project

This project is a **YOLOv8-based object detection pipeline** with **GitHub Actions CI/CD** and extensions for **AIOps**.

## 📂 Project Structure

## 🚀 Features
- Object detection with **YOLOv11 (Ultralytics)**
- **CI/CD with GitHub Actions**
  - Runs tests on every push/PR
  - Verifies YOLO model inference
- **AIOps Ready**
  - Example retraining pipeline (`retrain.yml`)
  - Save trained weights as artifacts
  - Extendable with monitoring & alerts

## ⚙️ Setup
Install dependencies:

```bash
pip install -r requirements.txt
python train.py
python detect.py
pytest -v
