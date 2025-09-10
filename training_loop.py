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

    print(model)

    train_vae(
        model, dataloader, dataloader, epochs=10, lr=1e-3, beta=1.0, device=device
    )

    print("Training complete.")

    # save model
    torch.save(model.state_dict(), "bottle_large.pt2")

    # anomaly_dataset = H5Dataset(dataset_name, "bottle", "broken_large")

    # anomaly_dataloader = DataLoader(anomaly_dataset, batch_size=32, shuffle=False)


if __name__ == "__main__":

    main()
