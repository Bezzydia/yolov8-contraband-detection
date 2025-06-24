import os
import zipfile
from ultralytics import YOLO
from pathlib import Path

DATASET_ZIP = "Contraband detection.v1i.yolov8.zip"
DATASET_DIR = Path("dataset")
MODEL_NAME = "yolov8s.pt"
EPOCHS = 50
BATCH_SIZE = 8
IMG_SIZE = 640

def validate_dataset_structure():
    required_paths = [
        DATASET_DIR / "data.yaml",
        DATASET_DIR / "train/images",
        DATASET_DIR / "train/labels",
        DATASET_DIR / "valid/images",
        DATASET_DIR / "valid/labels"
    ]
    
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(f"Ошибка структуры датасета. Отсутствуют:\n{'\n'.join(missing)}")
        return False
    
    # Проверка наличия файлов
    train_images = list((DATASET_DIR / "train/images").glob("*"))
    if not train_images:
        print("Ошибка: Нет изображений в train/images")
        return False
        
    return True

def main():
    # Распаковка датасета
    if not Path(DATASET_ZIP).exists():
        print(f"Ошибка: файл {DATASET_ZIP} не найден!")
        return
    
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    
    # Валидация структуры
    if not validate_dataset_structure():
        return
    
    # Обучение с абсолютными путями
    data_yaml = DATASET_DIR / "data.yaml"
    model = YOLO(MODEL_NAME)
    
    model.train(
        data=str(data_yaml.resolve()),  # Абсолютный путь
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="contraband_model",
        exist_ok=True  # Перезаписывает существующие результаты
    )

if __name__ == "__main__":
    main()