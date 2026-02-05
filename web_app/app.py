import streamlit as st
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

import config
from detect import HelmetDetector
import utils

st.set_page_config(
    page_title="Helmet Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css(Path("web_app/static/css/App.css"))

@st.cache_resource
def load_detector(model_path=None, conf_threshold=None):
    return HelmetDetector(model_path=model_path, conf_threshold=conf_threshold)


def main():
    st.markdown("<h1>Helmet Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #e0e0e0; font-size: 1.2rem;'>AI-Powered Safety Monitoring for Two-Wheeler Riders</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        model_option = st.selectbox(
            "Model",
            ["Trained Model", "Pre-trained YOLOv8n", "Pre-trained YOLOv8s"],
            help="Select which model to use for detection"
        )
        
        if model_option == "Trained Model":
            model_path = config.TRAINED_MODEL
        elif model_option == "Pre-trained YOLOv8n":
            model_path = "yolov8n.pt"
        else:
            model_path = "yolov8s.pt"
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Detection Classes")
        st.markdown("🟢 **With Helmet** - Safe")
        st.markdown("🔴 **Without Helmet** - Unsafe")
    
    try:
        detector = load_detector(model_path=model_path, conf_threshold=conf_threshold)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model weights are available or use a pre-trained model.")
        return
    
    tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])
    
    with tab1:
        st.markdown("### Upload an Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of a two-wheeler rider"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(image, use_container_width=True)
            
            if st.button("🔍 Detect Helmets", key="detect_image"):
                with st.spinner("Detecting..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        
                        start_time = time.time()
                        annotated_img, detections = detector.detect_image(
                            tmp_file.name,
                            show=False
                        )
                        inference_time = time.time() - start_time
                    
                    with col2:
                        st.markdown("#### Detection Results")
                        st.image(annotated_img, use_container_width=True)
                    
                    st.markdown("---")
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(detections)}</div>
                            <div class="metric-label">Total Detections</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    class_counts = {}
                    for det in detections:
                        class_name = det['class']
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    with metric_cols[1]:
                        count = class_counts.get('with_helmet', 0)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #4ade80;">{count}</div>
                            <div class="metric-label">With Helmet</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        count = class_counts.get('without_helmet', 0)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #f87171;">{count}</div>
                            <div class="metric-label">Without Helmet</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{inference_time:.2f}s</div>
                            <div class="metric-label">Inference Time</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if detections:
                        st.markdown("### 📋 Detection Details")
                        for i, det in enumerate(detections, 1):
                            st.markdown(f"""
                            **Detection {i}:**
                            - Class: `{det['class']}`
                            - Confidence: `{det['confidence']:.2%}`
                            - Bounding Box: `{[int(x) for x in det['bbox']]}`
                            """)
    
    with tab2:
        st.markdown("### Upload a Video")
        
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video of two-wheeler riders"
        )
        
        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            st.video(uploaded_video)
            
            if st.button("🔍 Process Video", key="detect_video"):
                with st.spinner("Processing video... This may take a while."):
                    output_path = config.RESULTS_DIR / f"detected_video_{int(time.time())}.mp4"
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        detector.detect_video(
                            video_path,
                            save_path=str(output_path),
                            show=False
                        )
                        
                        st.success(f"✓ Video processed successfully!")
                        st.info(f"Output saved to: {output_path}")
                        
                        if output_path.exists():
                            st.markdown("### Processed Video")
                            st.video(str(output_path))
                    
                    except Exception as e:
                        st.error(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
