import os
import sys
import time
from pathlib import Path
import tempfile
from io import BytesIO
import base64
import cv2
from PIL import Image
import numpy as np

from flask import Flask, render_template, request, Response, redirect, url_for, jsonify

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

import config
from detect import HelmetDetector
import utils

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

DETECTOR = None

def get_detector():
    global DETECTOR
    if DETECTOR is None:
        try:
            DETECTOR = HelmetDetector(model_path=config.TRAINED_MODEL, conf_threshold=0.25)
        except Exception as e:
            print(f"Failed to load configured model: {e}")
            DETECTOR = HelmetDetector(model_path="yolov8s.pt", conf_threshold=0.25)
    return DETECTOR


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect/image', methods=['GET', 'POST'])
def detect_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
            
        if file:
            try:
                img_bytes = file.read()
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name

                detector = get_detector()
                
                start_time = time.time()
                annotated_img, detections = detector.detect_image(tmp_path, show=False)
                inference_time = time.time() - start_time
                
                img_pil = Image.fromarray(annotated_img)
                buffered = BytesIO()
                img_pil.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                os.unlink(tmp_path)
                
                counts = {'with_helmet': 0, 'without_helmet': 0}
                for d in detections:
                    counts[d['class']] = counts.get(d['class'], 0) + 1
                    
                result_data = {
                    'image': img_str,
                    'detections': len(detections),
                    'with_helmet': counts['with_helmet'],
                    'without_helmet': counts['without_helmet'],
                    'time': f"{inference_time:.2f}s"
                }
                
                return jsonify(result_data)
                
            except Exception as e:
                return jsonify({'error': str(e)})
                
    return render_template('image_detect.html')

active_video_path = None

@app.route('/detect/video', methods=['GET', 'POST'])
def detect_video():
    global active_video_path
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            file.save(tmp_video.name)
            active_video_path = tmp_video.name
            
            return render_template('video_detect.html', video_ready=True)
            
    return render_template('video_detect.html', video_ready=False)


@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


def generate_frames(source=0):
    detector = get_detector()
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        return
        
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = detector.model.predict(
            source=frame_rgb,
            conf=detector.conf_threshold,
            iou=detector.iou_threshold,
            verbose=False
        )
        
        detections = utils.parse_yolo_results(results[0])
        annotated_frame = utils.draw_detections(frame_rgb, detections)
        
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        cv2.putText(annotated_frame_bgr, f"Detections: {len(detections)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame_bgr)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/video_feed')
def video_feed():
    source_type = request.args.get('source', 'webcam')
    if source_type == 'upload':
        global active_video_path
        if not active_video_path or not os.path.exists(active_video_path):
            return "No video uploaded", 404
        return Response(generate_frames(source=active_video_path), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_frames(source=0), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    get_detector()
    app.run(host='0.0.0.0', port=5000, debug=True)
