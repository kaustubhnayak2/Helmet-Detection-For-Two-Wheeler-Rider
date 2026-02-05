import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset Paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Model Paths
# PRETRAINED_MODEL = MODELS_DIR / "yolov8n.pt"
PRETRAINED_MODEL = MODELS_DIR / "yolov8s.pt"
# PRETRAINED_MODEL = MODELS_DIR / "yolov8m.pt"
TRAINED_MODEL = MODELS_DIR / "best.pt"

# Class Configuration
CLASSES = {
    0: 'with_helmet',
    1: 'without_helmet'
}
NUM_CLASSES = len(CLASSES)

# Model Hyperparameters
MODEL_CONFIG = {
    'model_name': 'yolov8s.pt',
    'img_size': 640,
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'max_det': 300,
}

# Training Hyperparameters
TRAINING_CONFIG = {
    'epochs': 150,
    'batch_size': 16,
    'learning_rate': 0.01,
    'optimizer': 'SGD',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'patience': 50,
    'save_period': 10,
    'workers': 8,
    'device': '',
}

# Data Augmentation
AUGMENTATION_CONFIG = {
    'hsv_h': 0.0,
    'hsv_s': 0.35,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
}

# Detection Settings
DETECTION_CONFIG = {
    'save_results': True,
    'save_txt': True,
    'save_conf': True,
    'show_labels': True,
    'show_conf': True,
    'line_thickness': 3,
    'font_scale': 0.5,
}

# Video/Webcam Settings
VIDEO_CONFIG = {
    'fps': 30,
    'frame_skip': 1,
    'display_width': 1280,
    'display_height': 720,
}

# Color Scheme for Bounding Boxes (RGB format)
COLORS = {
    'with_helmet': (0, 255, 0),
    'without_helmet': (255, 0, 0),
}

# Dataset YAML Configuration (for YOLO training)
DATASET_YAML = {
    'path': str(DATA_DIR.absolute()),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'names': CLASSES,
    'nc': NUM_CLASSES,
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, RAW_DATA_DIR,
                  PROCESSED_DATA_DIR, ANNOTATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
