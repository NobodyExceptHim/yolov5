import os
import json
import shutil

# Paths
coco_annotations_path = './result.json'
images_dir = './images'
output_dir = './dataset'
train_split = 0.8  # Proporcja danych treningowych

# Create directories
os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

# Load COCO annotations
with open(coco_annotations_path) as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = {cat['id']: cat['name'] for cat in coco['categories']}

# Split data
train_images = images[:int(len(images) * train_split)]
val_images = images[int(len(images) * train_split):]

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_images(images, subset):
    for img in images:
        image_id = img['id']
        filename = img['file_name']
        width = img['width']
        height = img['height']
        img_path = os.path.join(images_dir, filename)

        # Copy image to the appropriate directory
        if subset == 'train':
            shutil.copy(img_path, os.path.join(output_dir, 'images', 'train', filename))
            label_path = os.path.join(output_dir, 'labels', 'train', filename.replace('.jpg', '.txt'))
        else:
            shutil.copy(img_path, os.path.join(output_dir, 'images', 'val', filename))
            label_path = os.path.join(output_dir, 'labels', 'val', filename.replace('.jpg', '.txt'))

        # Process annotations
        with open(label_path, 'w') as label_file:
            for ann in annotations:
                if ann['image_id'] == image_id:
                    category_id = ann['category_id']
                    category_name = categories[category_id]
                    bbox = convert_bbox((width, height), ann['bbox'])
                    label_file.write(f"{category_id - 1} {' '.join(map(str, bbox))}\n")

# Process training and validation images
process_images(train_images, 'train')
process_images(val_images, 'val')

print("Conversion completed.")
