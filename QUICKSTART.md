# Quick Start Guide - Helmet Detection System

## 🚀 5-Minute Quick Start

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd helmet_detection_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Demo (1 minute)

```bash
# Run the demo script
python demo.py
```

This will:
- Load the pre-trained YOLOv8 model
- Create sample instructions
- Show you next steps

### Step 3: Test with Your Images (2 minutes)

```bash
# Option A: Single image
python src/detect.py --source path/to/your/image.jpg

# Option B: Webcam (real-time)
python src/detect.py --source 0

# Option C: Web interface
streamlit run web_app/app.py
```

---

## 📝 Common Use Cases

### 1. Detect in Single Image

```bash
python src/detect.py --source test_image.jpg --save results/output.jpg
```

### 2. Detect in Video

```bash
python src/detect.py --source test_video.mp4 --save results/output.mp4
```

### 3. Real-time Webcam

```bash
python src/detect.py --source 0
# Press 'q' to quit
# Press 's' to save screenshot
```

### 4. Batch Process Multiple Images

```bash
python src/detect.py --source images_folder/ --save results/batch/
```

### 5. Web Interface

```bash
streamlit run web_app/app.py
# Open browser to http://localhost:8501
```

---

## 🎓 Training Your Own Model

### Step 1: Prepare Dataset

```bash
# Download dataset info
python src/prepare_dataset.py --action download-info

# After downloading, organize your dataset:
# data/
#   train/
#     images/
#     labels/
#   val/
#     images/
#     labels/
```

### Step 2: Split Dataset (if needed)

```bash
python src/prepare_dataset.py --action split \
  --images path/to/all/images \
  --labels path/to/all/labels \
  --output data/
```

### Step 3: Train Model

```bash
python src/train.py
```

Training will:
- Use GPU if available (much faster!)
- Save checkpoints every 10 epochs
- Save best model to `models/best.pt`
- Generate training plots

### Step 4: Validate Model

```bash
python src/train.py --validate --model models/best.pt
```

### Step 5: Use Trained Model

```bash
python src/detect.py --source test.jpg --weights models/best.pt
```

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

### Change Model Size

```python
MODEL_CONFIG = {
    'model_name': 'yolov8s.pt',  # n, s, m, l, x (larger = more accurate but slower)
}
```

### Adjust Detection Threshold

```python
MODEL_CONFIG = {
    'conf_threshold': 0.5,  # Higher = fewer but more confident detections
}
```

### Modify Training Parameters

```python
TRAINING_CONFIG = {
    'epochs': 200,        # More epochs = better training (if data is good)
    'batch_size': 32,     # Larger batch = faster training (needs more GPU memory)
}
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'ultralytics'"

**Solution:**
```bash
pip install ultralytics
```

### Issue: "CUDA out of memory"

**Solution:**
```python
# In config.py, reduce batch size:
TRAINING_CONFIG = {
    'batch_size': 8,  # or even 4
}
```

### Issue: "Model not found"

**Solution:**
```bash
# YOLOv8 will auto-download on first use
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
```

### Issue: Webcam not working

**Solution:**
```bash
# Try different camera IDs:
python src/detect.py --source 1  # or 2, 3, etc.
```

### Issue: Low FPS on webcam

**Solution:**
- Use GPU if available
- Use smaller model (yolov8n)
- Reduce image size in config.py

---

## 📊 Expected Performance

| Model | Speed (FPS) | Accuracy | Use Case |
|-------|-------------|----------|----------|
| YOLOv8n | ~30 FPS (GPU) | Good | Real-time applications |
| YOLOv8s | ~25 FPS (GPU) | Better | Balanced |
| YOLOv8m | ~20 FPS (GPU) | Best | High accuracy needed |

*FPS measured on NVIDIA RTX 3060*

---

## 🎯 Next Steps

1. ✅ **Test the system** with demo images
2. ✅ **Try webcam** for real-time detection
3. ✅ **Collect dataset** for your specific use case
4. ✅ **Train model** on your dataset
5. ✅ **Deploy** to production environment

---

## 📚 Additional Resources

- **Full Documentation**: See [README.md](README.md)
- **YOLOv8 Docs**: https://docs.ultralytics.com
- **Dataset Sources**: Run `python src/prepare_dataset.py --action download-info`
- **Issues**: Check GitHub issues or create new one

---

## 💡 Tips for Best Results

1. **Dataset Quality**
   - Collect diverse images (different angles, lighting, weather)
   - Minimum 1000+ images recommended
   - Balance classes (equal with/without helmet)

2. **Training**
   - Use GPU for faster training
   - Monitor validation loss (should decrease)
   - Stop if validation loss stops improving

3. **Detection**
   - Adjust confidence threshold based on use case
   - Use appropriate model size for your hardware
   - Test on various scenarios

4. **Deployment**
   - Export model to ONNX for production
   - Optimize for target hardware
   - Add error handling and logging

---

**Happy Detecting! 🪖**
