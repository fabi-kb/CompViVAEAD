## imports
import os
from preprocessing import *
from VAE import VAE
import torch
from torch.utils.data import DataLoader
from dataset import H5Dataset
from training import *
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage import io


def visualize_reconstructions(model, data_loader, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        data = next(iter(data_loader))
        data = data.to(device)
        recon_batch, _, _ = model(data)

        data = data.cpu()
        recon_batch = recon_batch.cpu()

        print(f"Recon image shape: {recon_batch[0].shape}")
        print(f"Org image shape: {data[0].shape}")
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        for i in range(num_samples):
            orig_img = data[i].permute(1, 2, 0).numpy()
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            recon_img = recon_batch[i].permute(1, 2, 0).numpy()
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # add workers
    num_workers = 8

    dataset_name = "mvtec_dataset.h5"

    input_channels = 3
    latent_dim = 1024
    hidden_layers = [32, 64, 128, 256]

    dataset = H5Dataset(dataset_name, "bottle", "train")
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=num_workers
    )

    model = VAE(
        input_channels=input_channels,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        image_size=128,
    ).to(device)

    model.load_state_dict(torch.load("bottle_large.pt2"))

    visualize_reconstructions(
        model=model, data_loader=dataloader, device=device, num_samples=5
    )

    # Estimate Threshold:
    errors = []
    model.eval()
    with torch.no_grad():
        for _, original in enumerate(dataloader):
            original.cpu()

            reconstructed, _, _ = model(original)

            err = F.mse_loss(reconstructed, original, reduction="mean")

            errors.append(err)

    # Arbitrary chosen, need to check what a proper method is!
    threshold = np.array(errors).max()

    print(f"Threshold: {threshold}")

    anomaly_dataset = H5Dataset(dataset_name, "bottle", "test", "broken_large")

    anomaly_dataloader = DataLoader(
        anomaly_dataset, batch_size=32, shuffle=True, num_workers=num_workers
    )

    recons_errors, anomalies, threshold = detect_anomaly(
        model, anomaly_dataloader, device, threshold
    )

    print(f"Reconstruction errors: {recons_errors}")
    print(f"Anomalies detected: {anomalies}")
    print(f"Threshold: {threshold}")


if __name__ == "__main__":
    main()
