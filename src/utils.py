import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from typing import List, Tuple, Dict
import config


def load_image(image_path: str) -> np.ndarray:

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, output_path: str):

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)


def draw_detections(image: np.ndarray, detections: List[Dict], 
                    show_conf: bool = True) -> np.ndarray:
  
    img = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        color = config.COLORS.get(class_name, (255, 255, 255))
        
        color_bgr = (color[2], color[1], color[0])
        
        x1, y1, x2, y2 = map(int, bbox)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 
                     config.DETECTION_CONFIG['line_thickness'])
        
        if show_conf:
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = config.DETECTION_CONFIG['font_scale']
        thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness)
        
        cv2.rectangle(img_bgr, (x1, y1 - label_height - baseline - 5),
                     (x1 + label_width, y1), color_bgr, -1)
        
        cv2.putText(img_bgr, label, (x1, y1 - baseline - 5),
                   font, font_scale, (255, 255, 255), thickness)
        
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img


def parse_yolo_results(results) -> List[Dict]:

    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            
            if class_id not in config.CLASSES:
                continue
            
            detection = {
                'bbox': box.xyxy[0].cpu().numpy(),
                'confidence': float(box.conf[0]),
                'class_id': class_id,
                'class': config.CLASSES[class_id]
            }
            detections.append(detection)
    
    return detections


def calculate_metrics(predictions: List[Dict], ground_truth: List[Dict],
                     iou_threshold: float = 0.5) -> Dict:

    tp = 0 
    fp = 0 
    fn = 0  
    
    matched_gt = set()
    
    for pred in predictions:
        matched = False
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou >= iou_threshold and pred['class'] == gt['class']:
                tp += 1
                matched = True
                matched_gt.add(i)
                break
        
        if not matched:
            fp += 1
    
    fn = len(ground_truth) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def create_dataset_yaml(output_path: str):
    with open(output_path, 'w') as f:
        yaml.dump(config.DATASET_YAML, f, default_flow_style=False)


def plot_training_history(history_path: str, save_path: str = None):

    import pandas as pd
    
    df = pd.read_csv(history_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[0, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision & Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].set_title('Mean Average Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    axes[1, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def get_video_writer(output_path: str, fps: int, frame_size: Tuple[int, int]):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)