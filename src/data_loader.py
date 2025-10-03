import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .config import *

class H5Dataset(Dataset):
    def __init__(self, h5_path, category, dataset_type, defect_type=None):
        self.h5_path = h5_path
        self.category = category
        self.dataset_type = dataset_type
        self.defect_type = defect_type

        with h5py.File(h5_path, "r") as h5f:
            if defect_type is None:
                self.defect_types = list(h5f[category][dataset_type].keys())
                self.length = 0
                for d_type in self.defect_types:
                    self.length += h5f[category][dataset_type][d_type].attrs[
                        "num_images"
                    ]
            else:
                self.defect_types = [defect_type]
                self.length = h5f[category][dataset_type][defect_type].attrs[
                    "num_images"
                ]


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as h5f:
            if self.defect_type is None:
                current_idx = 0
                for d_type in self.defect_types:
                    num_images = h5f[self.category][self.dataset_type][d_type].attrs["num_images"]
                    if idx < current_idx + num_images:
                        local_idx = idx - current_idx
                        img = h5f[self.category][self.dataset_type][d_type]["images"][local_idx]
                        break
                    current_idx += num_images
            else:
                img = h5f[self.category][self.dataset_type][self.defect_type]["images"][idx]

            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img


class MVTecDataset(Dataset):
    """Wrapper class for H5Dataset"""
    
    def __init__(self, class_name, split="train", defect_type=None):
        """
        class_name: MVTec class name (e.g., 'bottle', 'carpet', 'metal_nut')
        split: 'train' or 'test'
        defect_type: Specific defect type to load, or None for all
        """
        self.class_name = class_name
        self.split = split
        self.defect_type = defect_type
        
        # self.h5_path = DATA_PATH
        print(f"Dataset split: {split}")
        if split in ['cutout_synth', 'scratches_synth']:
            self.synthetic = True
            self.dataset = load_synthetic_dataset(class_name, split)

            if self.dataset is None:
                raise ValueError(f"{split} synthetic dataset not found for {class_name}")

        else:
            # regular MVTec data from H5
            self.synthetic = False
            self.h5_path = DATA_PATH

            if not os.path.exists(self.h5_path):
                raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

            self.dataset = H5Dataset(
                h5_path=self.h5_path,
                category=class_name,
                dataset_type=split,
                defect_type=defect_type
            )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def load_synthetic_dataset(class_name, defect_type):
    """
    Load a synthetic defect dataset.
    
    class_name: MVTec class name
    defect_type: 'cutout_synth' or 'scratches_synth'

    returning dataset that has synthetic defect images
    """
    assert defect_type in ['cutout_synth', 'scratches_synth'], \
        f"Defect type must be 'cutout_synth' or 'scratches_synth', got {defect_type}"
    
    # synthetic_data_path = os.path.join(synthetic_data_path, 'synthetic', class_name, defect_type)
    synthetic_data_path = f"data/synthetic/{class_name}/{defect_type}"
    print(f"Looking for synthetic data in {synthetic_data_path}")

    if not os.path.exists(synthetic_data_path):
        raise ValueError(f"Synthetic data path {synthetic_data_path} does not exist")

    # get all PNG images in the directory
    image_files = [f for f in os.listdir(synthetic_data_path) if f.endswith('.png')]

    if len(image_files) == 0:
        raise ValueError(f"No synthetic images found in {synthetic_data_path}")

    print(f"Found {len(image_files)} synthetic images in {synthetic_data_path}")

    return SyntheticDataset(synthetic_data_path, image_files)


class SyntheticDataset(Dataset):
    """Dataset for synthetic defect images"""
    def __init__(self, data_path, image_files):
        self.data_path = data_path
        self.image_files = image_files
        
        # common transform for synthetic images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # we might need normalization to match MVTec data
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image
