import os
import numpy as np
from CompViVAEAD.src.config import DATA_PATH
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py


def generate_cutout(image, min_size=0.05, max_size=0.3, fill_value=None):
    """
    generate a cutout/occlusion defect.
    
    min_size: Minimum size as fraction of image
    max_size: Maximum size as fraction of image
    fill_value: Value to fill cutout with (None = use image mean)

    image with cutout defect is getting returned
    """
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
    else:
        pil_image = image.copy()
    
    width, height = pil_image.size
    
    # cutout size
    size_factor = random.uniform(min_size, max_size)
    cutout_width = int(width * size_factor)
    cutout_height = int(height * size_factor)
    
    # random position
    x = random.randint(0, width - cutout_width)
    y = random.randint(0, height - cutout_height)
    
    # fill color
    if fill_value is None:
        if isinstance(image, np.ndarray):
            fill_value = int(np.mean(image) * 255)
        else:
            fill_value = int(np.mean(np.array(pil_image)))
    
    # we draw a rectangle
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([x, y, x + cutout_width, y + cutout_height], fill=fill_value)
    
    if isinstance(image, np.ndarray):
        return np.array(pil_image).astype(np.float32) / 255.0
    return pil_image


def generate_scratches(image, num_lines=None, intensity=None, width=None):
    """
    generate scratches/lines defects.
    
    num_lines: Number of lines to draw (1-3 if None)
    intensity: Intensity of the lines (0-255, None = random)
    width: Width of the lines (1-5, None = random)
        
    image with scratch defects is getting returned
    """
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
    else:
        pil_image = image.copy()
    
    if num_lines is None:
        num_lines = random.randint(1, 3)
    
    if intensity is None:
        intensity = random.randint(180, 255)
    
    if width is None:
        width = random.randint(1, 5)
    
    width, height = pil_image.size
    draw = ImageDraw.Draw(pil_image)
    
    for _ in range(num_lines):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        
        # we drraw a line
        draw.line((x1, y1, x2, y2), fill=intensity, width=width)
    
    if isinstance(image, np.ndarray):
        return np.array(pil_image).astype(np.float32) / 255.0
    
    return pil_image


def create_synthetic_dataset_dirs(class_names):
    """
    we are creating directory structure for each synthetic data type.
    it will be neat this way
    
    class_names: list of class names
    """
    for class_name in class_names:
        cutout_dir = f"data/synthetic/{class_name}/cutout_synth"
        scratches_dir = f"data/synthetic/{class_name}/scratches_synth"
        
        os.makedirs(cutout_dir, exist_ok=True)
        os.makedirs(scratches_dir, exist_ok=True)
        
        print(f"Created directory: {cutout_dir}")
        print(f"Created directory: {scratches_dir}")

        
def generate_synthetic_dataset(class_name, source_h5_path, defect_type='cutout_synth', num_samples=100):
    """
    generate synthetic defect dataset from H5 file.
    
    class_name: MVTec class name
    defect_type: 'cutout_synth' or 'scratches_synth'
        
    (saves images to data/synthetic/<class_name>/<defect_type>/)
    """
    assert defect_type in ['cutout_synth', 'scratches_synth'], \
        f"Defect type must be 'cutout_synth' or 'scratches_synth', got {defect_type}"
    
    output_dir = f"data/synthetic/{class_name}/{defect_type}"
    os.makedirs(output_dir, exist_ok=True)
    

    with h5py.File(source_h5_path, 'r') as f:
        if class_name not in f:
            raise ValueError(f"Class {class_name} not found in H5 file")
        
        # test images
        test_path = f"{class_name}/test"
        test_defect_types = [key for key in f[test_path].keys()]
        
        all_images = []
        all_paths = []
        
        # get all 'good' test images
        if 'good' in f[test_path]:
            if 'images' in f[f"{test_path}/good"]:
                images = f[f"{test_path}/good/images"][:]
                all_images.extend(images)
                paths = [f"{defect_type}/good_{i}.png" for i in range(len(images))]
                all_paths.extend(paths)
        
        # if we need more samples, add from other defect types
        if len(all_images) < num_samples:
            for defect in test_defect_types:
                if defect == 'good':
                    continue
                    
                if 'images' in f[f"{test_path}/{defect}"]:
                    images = f[f"{test_path}/{defect}/images"][:]
                    needed = min(len(images), num_samples - len(all_images))
                    all_images.extend(images[:needed])
                    paths = [f"{defect_type}/{defect}_{i}.png" for i in range(needed)]
                    all_paths.extend(paths)
                    
                if len(all_images) >= num_samples:
                    break
        
        print(f"Generating {len(all_images)} synthetic {defect_type} images for {class_name}...")
        
        # apply defects and save
        for i, (image, path) in enumerate(tqdm(zip(all_images[:num_samples], all_paths[:num_samples]))):
            if defect_type == 'cutout_synth':
                defect_image = generate_cutout(image)
            else:
                defect_image = generate_scratches(image)
            
            output_path = os.path.join(output_dir, f"{i:04d}.png")
            
            # convert to uint8
            if isinstance(defect_image, np.ndarray):
                if defect_image.max() <= 1.0:
                    defect_image = (defect_image * 255).astype(np.uint8)
                Image.fromarray(defect_image).save(output_path)
            else:
                defect_image.save(output_path)
        
        print(f"Saved {num_samples} synthetic images to {output_dir}")


# We can remove this function if not needed (check)
def verify_synthetic_data_structure(class_names):
    """
    Verify synthetic data directory structure.
    
    Args:
        class_names: List of class names
    """
    for class_name in class_names:
        cutout_dir = f"data/synthetic/{class_name}/cutout_synth"
        scratches_dir = f"data/synthetic/{class_name}/scratches_synth"
        
        print(f"\nChecking structure for class: {class_name}")
        
        if not os.path.exists(cutout_dir):
            print(f"Warning: {cutout_dir} does not exist!")
        else:
            file_count = len([f for f in os.listdir(cutout_dir) if f.endswith('.png')])
            print(f"Found directory: {cutout_dir} with {file_count} images")
            
        if not os.path.exists(scratches_dir):
            print(f"Warning: {scratches_dir} does not exist!")
        else:
            file_count = len([f for f in os.listdir(scratches_dir) if f.endswith('.png')])
            print(f"Found directory: {scratches_dir} with {file_count} images")


if __name__ == "__main__":
    # example usage
    class_names = ["metal_nut", "carpet"]
    
    # Create directory structure
    create_synthetic_dataset_dirs(class_names)
    
    # Generate synthetic data
    for class_name in class_names:
        # Replace with your H5 path
        h5_path = DATA_PATH
        
        generate_synthetic_dataset(class_name, h5_path, defect_type='cutout_synth', num_samples=50)
        
        generate_synthetic_dataset(class_name, h5_path, defect_type='scratches_synth', num_samples=50)
    
    verify_synthetic_data_structure(class_names)
