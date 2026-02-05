import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import time

sys.path.append(str(Path(__file__).parent))
import config
import utils


class HelmetDetector:
    
    def __init__(self, model_path: str = None, conf_threshold: float = None):

        if model_path is None:
            model_path = config.TRAINED_MODEL
            if not Path(model_path).exists():
                print(f"Trained model not found at {model_path}")
                print("Using pre-trained YOLOv8 model instead...")
                model_path = config.MODEL_CONFIG['model_name']
        
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold or config.MODEL_CONFIG['conf_threshold']
        self.iou_threshold = config.MODEL_CONFIG['iou_threshold']
        
        print(f"Model loaded: {model_path}")
        print(f"Confidence threshold: {self.conf_threshold}")
    
    def detect_image(self, image_path: str, save_path: str = None, 
                    show: bool = False) -> tuple:

        img = utils.load_image(image_path)
        
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = utils.parse_yolo_results(results)
        
        annotated_img = utils.draw_detections(
            img, detections, 
            show_conf=config.DETECTION_CONFIG['show_conf']
        )
        
        if save_path:
            utils.save_image(annotated_img, save_path)
            print(f"Result saved to {save_path}")
        
        if show:
            self._display_image(annotated_img, "Helmet Detection")
        
        self._print_detection_summary(detections)
        
        return annotated_img, detections
    
    def detect_video(self, video_path: str, save_path: str = None, 
                    show: bool = True) -> None:

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print()
        
        writer = None
        if save_path:
            writer = utils.get_video_writer(save_path, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % config.VIDEO_CONFIG['frame_skip'] != 0:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = self.model.predict(
                    source=frame_rgb,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                detections = utils.parse_yolo_results(results)
                annotated_frame = utils.draw_detections(frame_rgb, detections)
                
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(annotated_frame_bgr, f"FPS: {current_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if writer:
                    writer.write(annotated_frame_bgr)
                
                if show:
                    cv2.imshow('Helmet Detection', annotated_frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | FPS: {current_fps:.1f}", end='\r')
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n✓ Processed {frame_count} frames")
            if save_path:
                print(f"✓ Result saved to {save_path}")
    
    def detect_webcam(self, camera_id: int = 0, save_path: str = None) -> None:

        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_CONFIG['display_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_CONFIG['display_height'])
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nWebcam started (Resolution: {width}x{height})")
        print("Press 'q' to quit, 's' to save screenshot")
        
        writer = None
        if save_path:
            writer = utils.get_video_writer(save_path, config.VIDEO_CONFIG['fps'], 
                                           (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = self.model.predict(
                    source=frame_rgb,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                detections = utils.parse_yolo_results(results)
                annotated_frame = utils.draw_detections(frame_rgb, detections)
                
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(annotated_frame_bgr, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame_bgr, f"Detections: {len(detections)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if writer:
                    writer.write(annotated_frame_bgr)
                
                cv2.imshow('Helmet Detection - Webcam', annotated_frame_bgr)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = config.RESULTS_DIR / f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(screenshot_path), annotated_frame_bgr)
                    print(f"\n✓ Screenshot saved to {screenshot_path}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\nWebcam stopped after {frame_count} frames")
    
    def detect_batch(self, input_dir: str, output_dir: str = None) -> None:

        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"\nFound {len(image_files)} images")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = config.RESULTS_DIR / "batch_detections"
            output_path.mkdir(parents=True, exist_ok=True)
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\nProcessing [{i}/{len(image_files)}]: {img_file.name}")
            
            save_path = output_path / img_file.name
            self.detect_image(str(img_file), str(save_path), show=False)
        
        print(f"\nBatch processing complete!")
        print(f"Results saved to {output_path}")
    
    def _display_image(self, image: np.ndarray, window_name: str = "Detection"):

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, img_bgr)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _print_detection_summary(self, detections: list):

        print(f"\nDetections: {len(detections)}")
        
        class_counts = {}
        for det in detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Helmet Detection System')
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image path, video path, directory, or webcam (0)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=None,
                       help='Confidence threshold')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save results')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display results')
    
    args = parser.parse_args()
    
    detector = HelmetDetector(model_path=args.weights, conf_threshold=args.conf)
    
    source = args.source
    show = not args.no_show
    
    if source.isdigit():
        detector.detect_webcam(camera_id=int(source), save_path=args.save)
    elif Path(source).is_file():
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if Path(source).suffix.lower() in video_extensions:
            detector.detect_video(source, save_path=args.save, show=show)
        else:
            save_path = args.save or str(config.RESULTS_DIR / f"detection_{Path(source).name}")
            detector.detect_image(source, save_path=save_path, show=show)
    elif Path(source).is_dir():
        detector.detect_batch(source, output_dir=args.save)
    else:
        print(f"Error: Invalid source '{source}'")
        print("Source must be: image path, video path, directory, or webcam ID (0)")
