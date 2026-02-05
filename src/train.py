import sys
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

sys.path.append(str(Path(__file__).parent))
import config
import utils


def prepare_dataset():

    dataset_yaml_path = config.DATA_DIR / "dataset.yaml"
    utils.create_dataset_yaml(str(dataset_yaml_path))   
    print(f"Dataset configuration saved to {dataset_yaml_path}")
    return dataset_yaml_path


def train_model(data_yaml: str, resume: bool = False):

    print("\n" + "="*60)
    print("HELMET DETECTION MODEL TRAINING")
    print("="*60 + "\n")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Load pre-trained model
    model_name = config.MODEL_CONFIG.get('model_name', 'yolov8s.pt')
    print(f"Loading pre-trained model: {model_name}")
    model = YOLO(model_name)
    
    # Training parameters
    train_config = {
        'data': str(data_yaml),
        'epochs': config.TRAINING_CONFIG['epochs'],
        'batch': config.TRAINING_CONFIG['batch_size'],
        'imgsz': config.MODEL_CONFIG['img_size'],
        'lr0': config.TRAINING_CONFIG['learning_rate'],
        'optimizer': config.TRAINING_CONFIG['optimizer'],
        'momentum': config.TRAINING_CONFIG['momentum'],
        'weight_decay': config.TRAINING_CONFIG['weight_decay'],
        'warmup_epochs': config.TRAINING_CONFIG['warmup_epochs'],
        'patience': config.TRAINING_CONFIG['patience'],
        'save_period': config.TRAINING_CONFIG['save_period'],
        'workers': config.TRAINING_CONFIG['workers'],
        'device': device,
        'project': str(config.RESULTS_DIR),
        'name': 'helmet_detection_train',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'resume': resume,
        
        # Data augmentation
        'hsv_h': config.AUGMENTATION_CONFIG['hsv_h'],
        'hsv_s': config.AUGMENTATION_CONFIG['hsv_s'],
        'hsv_v': config.AUGMENTATION_CONFIG['hsv_v'],
        'degrees': config.AUGMENTATION_CONFIG['degrees'],
        'translate': config.AUGMENTATION_CONFIG['translate'],
        'scale': config.AUGMENTATION_CONFIG['scale'],
        'shear': config.AUGMENTATION_CONFIG['shear'],
        'perspective': config.AUGMENTATION_CONFIG['perspective'],
        'flipud': config.AUGMENTATION_CONFIG['flipud'],
        'fliplr': config.AUGMENTATION_CONFIG['fliplr'],
        'mosaic': config.AUGMENTATION_CONFIG['mosaic'],
        'mixup': config.AUGMENTATION_CONFIG['mixup'],
    }
    
    # Print configuration
    print("\nTraining Configuration:")
    print("-" * 60)
    for key, value in train_config.items():
        print(f"{key:20s}: {value}")
    print("-" * 60 + "\n")
    
    # Start training
    print("Starting training...\n")
    results = model.train(**train_config)
    
    # Save best model to models directory
    best_model_path = config.RESULTS_DIR / 'helmet_detection_train' / 'weights' / 'best.pt'
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, config.TRAINED_MODEL)
        print(f"\nBest model saved to {config.TRAINED_MODEL}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    return results


def validate_model(model_path: str = None):
    
    if model_path is None:
        model_path = config.TRAINED_MODEL
    
    print("\n" + "="*60)
    print("MODEL VALIDATION")
    print("="*60 + "\n")
    
    model = YOLO(str(model_path))
    dataset_yaml = config.DATA_DIR / "dataset.yaml"
    
    # Run validation
    metrics = model.val(data=str(dataset_yaml))
    
    print("\nValidation Results:")
    print("-" * 60)
    print(f"mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")
    print("-" * 60)
    
    return metrics


def export_model(model_path: str = None, format: str = 'onnx'):

    if model_path is None:
        model_path = config.TRAINED_MODEL
    
    print(f"\nExporting model to {format} format...")
    model = YOLO(str(model_path))
    model.export(format=format)
    print(f"Model exported successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Helmet Detection Model')
    parser.add_argument('--data', type=str, help='Path to dataset YAML file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--validate', action='store_true', help='Validate model only')
    parser.add_argument('--export', type=str, help='Export model format (onnx, torchscript, etc.)')
    parser.add_argument('--model', type=str, help='Path to model weights for validation/export')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_model(args.model)
    elif args.export:
        export_model(args.model, args.export)
    else:
        data_yaml = args.data if args.data else prepare_dataset()
        train_model(data_yaml, resume=args.resume)
        
        print("\nRunning validation on trained model...")
        validate_model()
