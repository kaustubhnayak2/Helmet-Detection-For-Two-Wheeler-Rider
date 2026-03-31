# 🪖 Helmet Detection System for Two-Wheeler Riders

An AI/ML-powered computer vision system that detects whether two-wheeler riders are wearing helmets using **YOLOv8** deep learning model. This system can process images, videos, and real-time webcam feeds to identify riders and classify helmet usage for road safety monitoring.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **Real-time Detection**: Process webcam feed with live helmet detection
- **Multi-source Support**: Works with images, videos, and webcam
- **High Accuracy**: YOLOv8-based detection with customizable confidence thresholds
- **Web Interface**: Beautiful Streamlit web app for easy demonstration
- **Batch Processing**: Process multiple images at once
- **GPU Acceleration**: CUDA support for faster inference
- **Flexible Configuration**: Easy-to-modify config file for all parameters
- **Training Pipeline**: Complete training workflow for custom datasets

## 📋 Detection Classes

The system detects three classes:
- 🟢 **with_helmet**: Rider wearing a helmet (Safe)
- 🔴 **without_helmet**: Rider not wearing a helmet (Unsafe)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for better performance)
- Webcam (for real-time detection)

### Installation

1. **Clone or download the project**
   ```bash
   cd helmet_detection_project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained weights** (optional)
   ```bash
   # YOLOv8 weights will be downloaded automatically on first use
   # Or manually download from: https://github.com/ultralytics/assets/releases
   ```

## 💻 Usage

### 1. Image Detection

Detect helmets in a single image:

```bash
python src/detect.py --source path/to/image.jpg --save results/output.jpg
```

**Example:**
```bash
python src/detect.py --source test_images/rider1.jpg --weights models/best.pt --conf 0.25
```

### 2. Video Detection

Process a video file:

```bash
python src/detect.py --source path/to/video.mp4 --save results/output.mp4
```

**Example:**
```bash
python src/detect.py --source test_video.mp4 --weights models/best.pt --no-show
```

### 3. Webcam Detection (Real-time)

Run real-time detection from webcam:

```bash
python src/detect.py --source 0
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot

**Save webcam recording:**
```bash
python src/detect.py --source 0 --save results/webcam_recording.mp4
```

### 4. Batch Processing

Process multiple images in a directory:

```bash
python src/detect.py --source path/to/images/ --save results/batch_output/
```

### 5. Web Application

Launch the interactive web interface:

```bash
python flask_app/app.py
```

Then open your browser to `http://localhost:5000`

**Features:**
- Upload and detect on images
- Process video files
- Adjust confidence threshold
- View detection statistics
- Beautiful glassmorphism UI

## 🎓 Training Your Own Model

### 1. Prepare Dataset

Organize your dataset in YOLO format:

```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Label format** (YOLO format - one line per object):
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1).

**Example:**
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.3
```

### 2. Configure Dataset

Edit `src/config.py` to point to your dataset:

```python
DATASET_YAML = {
    'path': str(DATA_DIR.absolute()),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'names': {0: 'with_helmet', 1: 'without_helmet', 2: 'rider'},
    'nc': 3,
}
```

### 3. Train Model

```bash
python src/train.py
```

**With custom parameters:**
```bash
python src/train.py --data data/dataset.yaml
```

**Resume training:**
```bash
python src/train.py --resume
```

**Training configuration** can be modified in `src/config.py`:
- Epochs: 100 (default)
- Batch size: 16
- Learning rate: 0.01
- Image size: 640x640

### 4. Validate Model

```bash
python src/train.py --validate --model models/best.pt
```

### 5. Export Model

Export to different formats for deployment:

```bash
# ONNX format
python src/train.py --export onnx --model models/best.pt

# TensorFlow Lite
python src/train.py --export tflite --model models/best.pt
```

## 📁 Project Structure

```
helmet_detection_project/
├── data/                      # Dataset directory
│   ├── raw/                   # Original images
│   ├── processed/             # Preprocessed images
│   ├── annotations/           # Annotation files
│   ├── train/                 # Training data
│   ├── val/                   # Validation data
│   └── test/                  # Test data
├── models/                    # Model weights
│   ├── yolov8n.pt            # Pre-trained weights
│   └── best.pt               # Trained model
├── src/                       # Source code
│   ├── config.py             # Configuration file
│   ├── train.py              # Training script
│   ├── detect.py             # Detection script
│   └── utils.py              # Utility functions
├── flask_app/                   # Web application
│   ├── app.py                # Streamlit app
│   ├── static/               # CSS, JS, images
│   └── templates/            # HTML templates
├── notebooks/                 # Jupyter notebooks
│   └── exploration.ipynb     # Data exploration
├── results/                   # Output results
│   └── detections/           # Detection outputs
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## ⚙️ Configuration

All configurations are in `src/config.py`:

### Model Parameters
```python
MODEL_CONFIG = {
    'model_name': 'yolov8n.pt',    # Model size
    'img_size': 640,                # Input size
    'conf_threshold': 0.25,         # Confidence threshold
    'iou_threshold': 0.45,          # NMS threshold
}
```

### Training Parameters
```python
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.01,
    'optimizer': 'SGD',
    'patience': 50,
}
```

### Detection Colors
```python
COLORS = {
    'with_helmet': (0, 255, 0),      # Green
    'without_helmet': (0, 0, 255),   # Red
    'rider': (255, 255, 0),          # Cyan
}
```

## 📊 Model Performance

Expected performance metrics (on well-prepared dataset):

| Metric | Value |
|--------|-------|
| mAP@0.5 | >85% |
| mAP@0.5:0.95 | >60% |
| Inference Speed (GPU) | ~30 FPS |
| Inference Speed (CPU) | ~10 FPS |

## 🎯 Use Cases

1. **Traffic Monitoring**: Automated helmet compliance checking at traffic signals
2. **Parking Enforcement**: Helmet detection in parking areas
3. **Safety Audits**: Analyzing helmet usage patterns
4. **Smart City Integration**: Real-time safety monitoring systems
5. **Research**: Road safety studies and statistics

## 📚 Dataset Resources

Public datasets for helmet detection:

1. **Roboflow Helmet Detection Dataset**
   - https://universe.roboflow.com/search?q=helmet%20detection

2. **Kaggle Datasets**
   - https://www.kaggle.com/datasets/andrewmvd/helmet-detection

3. **Custom Collection**
   - Use tools like LabelImg or Roboflow for annotation
   - Collect diverse images (different angles, lighting, weather)
   - Minimum 1000+ images recommended

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 framework
- **OpenCV** for computer vision tools
- **Flask** for web interface
- Open-source community for datasets and resources

---

**Made with ❤️ for Road Safety**

*Remember: Wearing a helmet can save lives! 🪖*
