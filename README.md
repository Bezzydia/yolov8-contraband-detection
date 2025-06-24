# yolov8-contraband-detection
YOLOv8 Combat Knife Detection: Precise identification of 4 knife types with 0.8 precision, 0.7 recall, and 0.776 mAP@0.5. Robust performance across classes with progressive metric stabilization.
# Contraband detection > 2024-05-01 6:20pm
https://universe.roboflow.com/artificial-inteligence-zaqdt/contraband-detection-nde29

Provided by a Roboflow user
License: CC BY 4.0


# Contraband Detection using YOLOv8

This project trains a YOLOv8 object detection model to identify contraband items in images. The implementation handles dataset preparation, model training, and validation using Ultralytics YOLOv8 framework.

## Features
- Automatic dataset extraction from ZIP archive
- Dataset structure validation
- Customizable training parameters (epochs, batch size, image size)
- Model training with progress tracking

## Requirements
- Python 3.8+
- Ultralytics YOLOv8 (`pip install ultralytics`)
- Required packages: `torch`, `zipfile`, `pathlib`

## Dataset Preparation
1. Place your dataset ZIP file in the project root with filename:  
   **`Contraband detection.v1i.yolov8.zip`**
2. Required dataset structure inside ZIP:


## Usage
1. Install dependencies:
```bash
pip install ultralytics

python train.py

DATASET_ZIP = "Contraband detection.v1i.yolov8.zip"  # Dataset filename
MODEL_NAME = "yolov8s.pt"                            # Pretrained model
EPOCHS = 50                                           # Training epochs
BATCH_SIZE = 8                                        # Batch size
IMG_SIZE = 640                                        # Input 
image size

Output
Dataset extraction complete
Dataset validation passed
Training YOLOv8 model for 50 epochs...
Epoch 1/50: 100%|████| 100/100 [01:30<00:00, 1.10it/s]
...
Results saved to runs/detect/contraband_model
