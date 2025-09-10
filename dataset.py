import h5py
from torch.utils.data import Dataset
import torch


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
