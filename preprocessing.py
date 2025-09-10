import numpy as np
import os
from skimage import io, transform, exposure, color
import sys
import h5py
import tqdm as tqdm


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
