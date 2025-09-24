import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
        # Get image from H5Dataset
        image = self.dataset[idx]
        
        # In MVTec convention, return a dict with 'image' key
        # This makes it more compatible with other datasets
        return image  # Or return {'image': image} if you prefer dict output


def load_synthetic_dataset(class_name, defect_type):
    """
    Load a synthetic defect dataset.
    
    class_name: MVTec class name
    defect_type: 'cutout_synth' or 'scratches_synth'

    returning dataset that  has synthetic defect images
    """
    assert defect_type in ['cutout_synth', 'scratches_synth'], \
        f"Defect type must be 'cutout_synth' or 'scratches_synth', got {defect_type}"
    
    data_path = f"data/synthetic/{class_name}/{defect_type}"
    
    if not os.path.exists(data_path):
        raise ValueError(f"Synthetic data path {data_path} does not exist")
    
    # get all PNG images in the directory
    image_files = [f for f in os.listdir(data_path) if f.endswith('.png')]
    
    if len(image_files) == 0:
        raise ValueError(f"No synthetic images found in {data_path}")
    
    print(f"Found {len(image_files)} synthetic images in {data_path}")
    
    # Load and preprocess images
    # ... implementation depends on your dataset class ...
    
    # Example assuming a PyTorch Dataset:
    # return SyntheticDataset(data_path, transform=transform)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as h5f:
            if self.defect_type is None:
                current_idx = 0
                for d_type in self.defect_types:
                    num_images = h5f[self.category][self.dataset_type][d_type].attrs[
                        "num_images"
                    ]
                    if idx < current_idx + num_images:
                        local_idx = idx - current_idx
                        img = h5f[self.category][self.dataset_type][d_type]["images"][
                            local_idx
                        ]
                        break
                    current_idx += num_images
            else:
                img = h5f[self.category][self.dataset_type][self.defect_type]["images"][
                    idx
                ]

            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img
