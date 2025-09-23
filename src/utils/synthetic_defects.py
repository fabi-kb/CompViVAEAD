import os
import random
import numpy as np
import h5py
import cv2
from PIL import Image
from tqdm import tqdm

SYNTHETIC_DATA_BASE = os.path.join('data', 'synthetic')


def create_synthetic_dataset_dirs(class_names):
    """
    create directories for synthetic datasets for specified classes.
    """
    for class_name in class_names:
        class_dir = os.path.join(SYNTHETIC_DATA_BASE, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # defect type directories
        cutout_dir = os.path.join(class_dir, 'cutout_synth')
        scratches_dir = os.path.join(class_dir, 'scratches_synth')
        
        os.makedirs(cutout_dir, exist_ok=True)
        os.makedirs(scratches_dir, exist_ok=True)
        
        print(f"Created directories for {class_name}:")
        print(f"  - {cutout_dir}")
        print(f"  - {scratches_dir}")


def verify_synthetic_data_structure(class_names):
    """
    verify that synthetic data directories have been created and contain files.
    """
    for class_name in class_names:
        class_dir = os.path.join(SYNTHETIC_DATA_BASE, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} does not exist")
            continue
        
        # checking defect type directories and count files
        for defect_type in ['cutout_synth', 'scratches_synth']:
            defect_dir = os.path.join(class_dir, defect_type)
            if not os.path.exists(defect_dir):
                print(f"Warning: Defect directory {defect_dir} does not exist")
                continue
                
            files = [f for f in os.listdir(defect_dir) if f.endswith('_synth.png')]
            print(f"Verified {class_name}/{defect_type}: {len(files)} files")


def generate_synthetic_dataset(class_name, h5_path, defect_type, num_samples=50):
    """
    generate synthetic defects for a specific class and defect type.
    
    class_name: MVTec class name
    h5_path
    defect_type: cutout_synth or scratches_synth
    num_samples: # of synthetic samples to generate
    """
    # source images from H5 file
    with h5py.File(h5_path, 'r') as h5f:
        # Use 'train/good' as source for synthetic defects
        src_path = f"{class_name}/train/good/images"
        if src_path not in h5f:
            raise ValueError(f"Path {src_path} not found in H5 file")
        
        # all source images
        images = h5f[src_path][:]
        num_images = len(images)
        
        if num_images == 0:
            raise ValueError(f"No images found at {src_path}")
        
        print(f"Found {num_images} source images for {class_name}")
        
        output_dir = os.path.join(SYNTHETIC_DATA_BASE, class_name, defect_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # generate
        for i in tqdm(range(num_samples), desc=f"Generating {defect_type}"):
            # a random source image
            img_idx = random.randint(0, num_images-1)
            img = images[img_idx]
            
            # convert to RGB if grayscale
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # apply defect
            if defect_type == 'cutout_synth':
                defect_img = apply_cutout_defect(img.copy())
            else:
                defect_img = apply_scratch_defect(img.copy())

            # PIL threw an error when saving float images, we try the following
            # Convert to uint8 and ensure correct range before saving
            if defect_img.dtype != np.uint8:
                # If floating point, scale to 0-255 and convert to uint8
                if np.issubdtype(defect_img.dtype, np.floating):
                    defect_img = (defect_img * 255).clip(0, 255).astype(np.uint8)
                else:
                    defect_img = defect_img.astype(np.uint8)

            # save
            output_path = os.path.join(output_dir, f"{i:04d}_synth.png")
            Image.fromarray(defect_img).save(output_path)


def apply_cutout_defect(image):
    """
    appplying cutout defect to image
    we randomly remove a rectangular area (5-30% of image) and fill with 0 or image mean.
    
    image: Source image as numpy array
        
    return the image with cutout defect
    """
    h, w = image.shape[:2]
    
    # cutout size (5-30%)
    area_percent = random.uniform(0.05, 0.30)
    area_pixels = int(h * w * area_percent)
    
    # rectangle dimensions
    aspect_ratio = random.uniform(0.5, 2.0)
    rect_h = int(np.sqrt(area_pixels / aspect_ratio))
    rect_w = int(area_pixels / rect_h)
    
    rect_h = min(rect_h, h - 2)
    rect_w = min(rect_w, w - 2)
    
    x = random.randint(0, w - rect_w)
    y = random.randint(0, h - rect_h)
    
    # fill type (0=black or 1=mean)
    fill_type = random.randint(0, 1)
    
    if fill_type == 0:
        # fill with black
        image[y:y+rect_h, x:x+rect_w] = 0
    else:
        # with mean color
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                image[y:y+rect_h, x:x+rect_w, c] = np.mean(image[:,:,c])
        else:
            image[y:y+rect_h, x:x+rect_w] = np.mean(image)
    
    return image


def apply_scratch_defect(image):
    """
    appplying scratch defect to image
    we add 1-3 thin lines with width (1-5px) and intensity
    
    image: Source image as numpy array
        
    
    return the image with scratch defect
    """
    h, w = image.shape[:2]
    
    # number of scratches (1-3)
    num_scratches = random.randint(1, 3)
    
    for _ in range(num_scratches):
        # random width
        width = random.randint(1, 5)

        # random intensity (brightness of scratch)
        intensity = random.randint(180, 255)
        
        # random line parameters
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
        
        # draw
        cv2.line(
            image, 
            (x1, y1), 
            (x2, y2), 
            (intensity, intensity, intensity), 
            width
        )
    
    return image
