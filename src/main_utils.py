import argparse
import numpy as np
import os
import sys
import h5py
import tqdm as tqdm
from typing import Dict, Optional
from skimage import io, transform, exposure, color

from .config import BETA, CLASS_STATS, DATA_PATH, LAMBDA_GEO


def transform_image(
    image, target_size=(128, 128), normalize="0-1", grayscale=False, antialiasing=True
):
    """
    Transforms an image by resizing, normalizing, and converting to grayscale if specified.
    Parameters:
    - image: Input image as a NumPy array.
    - target_size: Tuple specifying the desired image size (height, width).
    - normalize: Normalization method ('0-1' for scaling to [0, 1], 'std' for standardization).
    - grayscale: Boolean indicating whether to convert the image to grayscale.
    - antialiasing: Boolean indicating whether to apply antialiasing when resizing.
    Returns:
    - Transformed image as a NumPy array.
    """

    image = transform.resize(image, target_size, anti_aliasing=antialiasing)

    if grayscale and len(image.shape) == 3:
        image = color.rgb2gray(image)

        image = np.expand_dims(image, axis=-1)

    if normalize == "0-1" and image.max() > 1:
        image = image.astype(np.float32) / 255
    if normalize == "std" and image.std() > 0:
        image = (image - np.mean(image)) / np.std(image)

    return image


def create_mvtec_h5_dataset(
    base_path,
    h5_filename,
    target_size=(128, 128),
    normalize="0-1",
    grayscale=False,
    antialiasing=True,
):
    """
    Function to create the dataset from the downloaded MVTEC dataset.

    Parameters:
    - base_path: string to the folder containing the dataset
    - h5_filename: string for the target dataset file
    - target_size: tuple for the target image size

    for the last three parameters see the transform_image function
    """
    with h5py.File(h5_filename, "w") as h5f:

        h5f.attrs["description"] = "MVTec dataset"
        h5f.attrs["target_size"] = target_size
        h5f.attrs["normalize"] = normalize
        h5f.attrs["grayscale"] = grayscale
        h5f.attrs["antialiasing"] = antialiasing

        for category in os.listdir(base_path):
            category_path = os.path.join(base_path, category)
            if not os.path.isdir(category_path):
                continue

            category_group = h5f.create_group(category)

            for dataset_type in ["train", "test"]:
                dataset_path = os.path.join(category_path, dataset_type)
                if not os.path.isdir(dataset_path):
                    continue

                dataset_group = category_group.create_group(dataset_type)

                for defect_type in os.listdir(dataset_path):
                    defect_path = os.path.join(dataset_path, defect_type)
                    print(defect_path)
                    if not os.path.isdir(defect_path):
                        continue

                    defect_group = dataset_group.create_group(defect_type)
                    images = []

                    image_files = [
                        f
                        for f in os.listdir(defect_path)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]

                    for img_file in tqdm.tqdm(
                        image_files, desc=f"{category}/{dataset_type}/{defect_type}"
                    ):
                        img_path = os.path.join(defect_path, img_file)
                        img = io.imread(img_path)
                        transformed_img = transform_image(
                            img,
                            target_size=target_size,
                            normalize=normalize,
                            grayscale=grayscale,
                            antialiasing=antialiasing,
                        )
                        images.append(transformed_img)

                    if images:
                        defect_group.create_dataset(
                            "images",
                            data=np.array(images),
                            compression="gzip",
                            compression_opts=9,
                        )
                        defect_group.attrs["num_images"] = len(images)

        print("Dataset creation complete.")


# Small helper to compute per-class mean and std from HDF5 dataset
def compute_class_stats(
    h5_path: Optional[str] = None,
    classes: Optional[list] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, tuple]]:
    """
    Compute per-class channel mean/std from HDF5 layout like:
    <class>/train/good/images -> (N,H,W,3) or (N,H,W)

    Returns: CLASS_STATS dict with {class_name: {'mean': [...], 'std': [...]}}
    Note: This function loads images in memory in chunks; it tries to be memory efficient.
    """
    import h5py
    import numpy as np

    if h5_path is None:
        h5_path = DATA_PATH

    stats = {}
    with h5py.File(h5_path, "r") as f:
        all_classes = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        target_classes = classes if classes is not None else all_classes

        for cls in target_classes:
            try:
                ds = f[f"{cls}/train/good/images"]
            except Exception as e:
                # skip classes that don't match expected
                continue

            N = ds.shape[0]
            # optionally limit samples for speed
            count = min(N, max_samples) if max_samples else N

            # accumulate mean and var using Welford-like online formula across images
            sum_c = None
            sum_sq_c = None
            n_seen = 0

            for i in range(count):
                img = ds[i]  # shape (H,W,3) or (H,W)
                if img.ndim == 2:
                    # grayscale -> replicate to 3 channels for stats
                    img = np.repeat(img[:, :, None], 3, axis=2)
                # convert to float [0,1] if uint8
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                # sum across spatial dims -> per-channel sums
                if sum_c is None:
                    sum_c = img.sum(axis=(0, 1))
                    sum_sq_c = (img**2).sum(axis=(0, 1))
                else:
                    sum_c += img.sum(axis=(0, 1))
                    sum_sq_c += (img**2).sum(axis=(0, 1))
                n_seen += img.shape[0] * img.shape[1]

            # per-channel mean and std
            mean = (sum_c / n_seen).tolist()
            var = (sum_sq_c / n_seen) - (np.array(mean) ** 2)
            std = np.sqrt(np.maximum(var, 1e-8)).tolist()
            stats[cls] = {"mean": mean, "std": std, "samples_used": count}
    # populate global container
    CLASS_STATS.update(stats)
    return stats


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train baseline VAE on a given MVTec class"
    )

    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        help="MVTec class to train on (e.g., metal_nut, carpet)",
    )
    parser.add_argument(
        "--data_subset",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0)",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation split ratio (0.0-0.5)"
    )
    parser.add_argument(
        "--beta", type=float, default=BETA, help="Weight for KL divergence term"
    )
    # keep lambda_geo argument for API compatibility (ignored by baseline training)
    parser.add_argument(
        "--lambda_geo",
        type=float,
        default=LAMBDA_GEO,
        help="Weight for geometric prior term (ignored by baseline)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Custom run identifier (used for experiment folder naming)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint path to resume from (optional)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "."
    save = sys.argv[2] if len(sys.argv) > 2 else False

    target_size = (128, 128)
    normalize = "0-1"
    grayscale = False
    antialiasing = True

    if save:
        h5_filename = "mvtec_dataset.h5"
        create_mvtec_h5_dataset(
            folder_path,
            h5_filename,
            target_size=target_size,
            normalize=normalize,
            grayscale=grayscale,
            antialiasing=antialiasing,
        )


if __name__ == "__main__":
    main()
