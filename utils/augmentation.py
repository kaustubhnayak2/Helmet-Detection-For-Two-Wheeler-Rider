"""
Augmentation Script for 'without_helmet' Images
Generates additional images to balance the dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm

INPUT_DIR = Path("./data/Helmet")
OUTPUT_DIR = Path("./data/processed")  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUGMENT_PER_IMAGE = 1  

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),
    A.ToGray(p=1.0)
])

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [f for ext in image_extensions for f in INPUT_DIR.glob(f"*{ext}")]

print(f"Found {len(image_files)} images to augment.")

augmented_count = 0
for img_path in tqdm(image_files, desc="Augmenting images"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    for i in range(AUGMENT_PER_IMAGE):
        augmented = transform(image=img)['image']
        output_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
        cv2.imwrite(str(OUTPUT_DIR / output_name), augmented)
        augmented_count += 1

print(f"✓ Augmentation complete! Generated {augmented_count} images in {OUTPUT_DIR}")
