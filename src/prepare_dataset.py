import os
import shutil
from pathlib import Path
import random
import yaml
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO


def create_directory_structure(base_path: str):

    base = Path(base_path)
    
    directories = [
        base / 'train' / 'images',
        base / 'train' / 'labels',
        base / 'val' / 'images',
        base / 'val' / 'labels',
        base / 'test' / 'images',
        base / 'test' / 'labels',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at {base_path}")


def split_dataset(images_dir: str, labels_dir: str, output_dir: str,
                 train_ratio: float = 0.7, val_ratio: float = 0.2,
                 test_ratio: float = 0.1, seed: int = 42):

    random.seed(seed)
    
    images_path = Path(images_dir)
    image_files = list(images_path.glob('*.jpg')) + \
                  list(images_path.glob('*.jpeg')) + \
                  list(images_path.glob('*.png'))
    
    random.shuffle(image_files)
    
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    create_directory_structure(output_dir)
    
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} set ({len(files)} images)...")
        
        for img_file in tqdm(files):
            dst_img = output_path / split_name / 'images' / img_file.name
            shutil.copy(img_file, dst_img)
            
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = output_path / split_name / 'labels' / label_file.name
                shutil.copy(label_file, dst_label)
    
    print(f"\n Dataset split complete!")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")


def create_dataset_yaml(output_dir: str, yaml_path: str):

    dataset_config = {
        'path': str(Path(output_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'with_helmet',
            1: 'without_helmet',
        },
        'nc': 2
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset YAML created at {yaml_path}")


def validate_dataset(dataset_dir: str):
    
    dataset_path = Path(dataset_dir)
    
    print("\nValidating dataset...")
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists():
            issues.append(f"Missing directory: {images_dir}")
            continue
        
        if not labels_dir.exists():
            issues.append(f"Missing directory: {labels_dir}")
            continue
        
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.jpeg')) + \
                     list(images_dir.glob('*.png'))
        
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"\n{split.upper()} set:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        
        missing_labels = []
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                missing_labels.append(img_file.name)
        
        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)} images without labels")
            if len(missing_labels) <= 5:
                for name in missing_labels:
                    issues.append(f"  - {name}")
    
    if issues:
        print("\n Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n Dataset validation passed!")
        
        
def check_blank_label(label_dir: str):
    
    label_dir = Path(label_dir)
    
    empty = 0
    total = 0
    
    for f in os.listdir(label_dir):
            if f.endswith(".txt"):
                total += 1
                if os.path.getsize(os.path.join(label_dir, f)) == 0:
                    empty += 1
    
    print(f"Total labels: {total}")
    print(f"Empty labels: {empty}")
    print(f"Empty %: {(empty/total)*100:.2f}%")    


def clean_empty_label(image_dir: str, label_dir: str):
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    USELESS_IMG_DIR = "./data/empty_label/images"
    USELESS_LBL_DIR = "./data/empty_label/labels"

    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

    os.makedirs(USELESS_IMG_DIR, exist_ok=True)
    os.makedirs(USELESS_LBL_DIR, exist_ok=True)

    moved = 0

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "r") as f:
            content = f.read().strip()

        if content == "":
            base_name = os.path.splitext(label_file)[0]

            shutil.move(label_path, os.path.join(USELESS_LBL_DIR, label_file))

            image_found = False
            for ext in IMAGE_EXTENSIONS:
                img_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(img_path):
                    shutil.move(
                        img_path,
                        os.path.join(USELESS_IMG_DIR, base_name + ext)
                    )
                    image_found = True
                    break

            if not image_found:
                print(f" Image not found for: {base_name}")

            moved += 1
            print(f" Moved to useless: {base_name}")
    
    print(f"\n Done. Total moved: {moved}")


def rename_labels_images(image_dir: str, label_dir: str, name: str):
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    new_name = Path(name)
        
    VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    label_basenames = {os.path.splitext(f)[0] for f in label_files if f.endswith(".txt")}

    count = 1

    for img_file in image_files:
        name, ext = os.path.splitext(img_file)
        ext = ext.lower()

        if ext not in VALID_EXTENSIONS:
            continue

        if name not in label_basenames:
            print(f" No label for image: {img_file}")
            continue

        old_img_path = os.path.join(image_dir, img_file)
        old_label_path = os.path.join(label_dir, name + ".txt")
        
        new_img_name = f"{new_name}_{count}.jpg"
        new_label_name = f"{new_name}_{count}.txt"

        new_img_path = os.path.join(image_dir, new_img_name)
        new_label_path = os.path.join(label_dir, new_label_name)

        img = Image.open(old_img_path).convert("RGB")
        img.save(new_img_path, "JPEG")

        if old_img_path != new_img_path:
            os.remove(old_img_path)

        os.rename(old_label_path, new_label_path)

        print(f"Renamed: {img_file} + {name}.txt → {new_img_name} & {new_label_name}")

        count += 1

    print("\nDone! Images and labels are renamed & synchronized.")
    
    
def create_label_from_image(image_dir: str, output_dir: str, image_type: str):
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    image_type = image_type
    
    model = YOLO("yolov8n.pt")
    
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)

        results = model(img_path, conf=0.4)

        for r in results:
            if r.boxes is None:
                continue

            h, w = r.orig_img.shape[:2]
            label_path = Path(output_dir) / f"{Path(img_name).stem}.txt"

            with open(label_path, "w") as f:
                for box in r.boxes:
                    if int(box.cls[0]) != 0:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0]
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    if (image_type == 'with_helmet'):
                        f.write(f"0 {xc} {yc} {bw} {bh}\n")
                    
                    elif (image_type == 'without_helmet'):
                        f.write(f"1 {xc} {yc} {bw} {bh}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Preparation Tool')
    parser.add_argument('--action', type=str, required=True,
                       choices=['create', 'split', 'validate', 'validate_label', 'clean_label', 'rename', 'label_creation'],
                       help='Action to perform')
    parser.add_argument('--images', type=str, help='Images directory')
    parser.add_argument('--labels', type=str, help='Labels directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--name', type=str, help='Name for Rename')
    parser.add_argument('--type', type=str, choices=['with_helmet', 'without_helmet'], help='Type Of Label')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio')
    
    args = parser.parse_args()
    
    if args.action == 'create':
        if not args.output:
            print("Error: --output required for create action")
        else:
            create_directory_structure(args.output)
            yaml_path = Path(args.output) / 'dataset.yaml'
            create_dataset_yaml(args.output, str(yaml_path))
    
    elif args.action == 'split':
        if not all([args.images, args.labels, args.output]):
            print("Error: --images, --labels, and --output required for split action")
        else:
            split_dataset(
                args.images, args.labels, args.output,
                args.train_ratio, args.val_ratio, args.test_ratio
            )
            yaml_path = Path(args.output) / 'dataset.yaml'
            create_dataset_yaml(args.output, str(yaml_path))
    
    elif args.action == 'validate':
        if not args.output:
            print("Error: --output required for validate action")
        else:
            validate_dataset(args.output)

    elif args.action == 'validate_label':
        if not args.labels:
            print("Error: --labels required for checking empty labels.")
        else:
            check_blank_label(args.labels)
            
    elif args.action == "clean_label":
        if not all([args.images, args.labels]):
            print("Error: --images, and --labels required for empty label cleaning action.")
        else:
            clean_empty_label(args.images, args.labels)
    
    elif args.action == "rename":
        if not all([args.images, args.labels, args.name]):
            print("Error: --images, --labels, and --name required for rename action.")
        else:
            rename_labels_images(args.images, args.labels, args.name)

    elif args.action == "label_creation":
        if not all([args.images, args.output, args.type]):
            print("Error: --images, --type and --output required for label creation action.")
        else:
            create_label_from_image(args.images, args.output, args.type)
            