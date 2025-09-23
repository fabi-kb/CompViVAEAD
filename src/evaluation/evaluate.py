## imports
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..main_utils import *
from ..models.vanilla_vae import VAE
from ..data_loader import H5Dataset
from ..config import EDGE_ON_DENORM, EDGE_GRAYSCALE
from ..training.losses import sobel_edges  # Import from wherever it's defined
from ..config import EDGE_ON_DENORM, EDGE_GRAYSCALE, CLASS_STATS, NORMALIZATION
#from training import *
import numpy as np
import h5py
import matplotlib.pyplot as plt
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


def denormalize(images, class_name=None):
    """
    Denormalizes images based on class statistics or fallback values.
    
    images: Tensor: [B, C, H, W]
    """
    if not EDGE_ON_DENORM:
        return images
        
    denorm_images = images.clone()
    
    if NORMALIZATION == 'per_class' and class_name is not None and class_name in CLASS_STATS:
        mean = torch.tensor(CLASS_STATS[class_name]['mean'], device=images.device).view(1, -1, 1, 1)
        std = torch.tensor(CLASS_STATS[class_name]['std'], device=images.device).view(1, -1, 1, 1)
        denorm_images = denorm_images * std + mean
    else:
        # imageNet fallback
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        denorm_images = denorm_images * std + mean
        
    return denorm_images

def compute_edge_score(original, recon, class_name=None):
    """
    Computes edge-based anomaly score between original and reconstructed images.
    
    original and recon: [B, C, H, W]
        
    we return per-image edge score tensor [B]
    """
    if EDGE_ON_DENORM:
        original = denormalize(original, class_name)
        recon = denormalize(recon, class_name)
    
    if EDGE_GRAYSCALE and original.shape[1] == 3:
        # RGB to grayscale weights
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=original.device).view(1, 3, 1, 1)
        original_gray = torch.sum(original * rgb_weights, dim=1, keepdim=True)
        recon_gray = torch.sum(recon * rgb_weights, dim=1, keepdim=True)
    else:
        original_gray = original
        recon_gray = recon
    
    # edge maps
    edges_original = sobel_edges(original_gray)
    edges_recon = sobel_edges(recon_gray)
    # per-image MSE
    edge_diff = F.mse_loss(edges_recon, edges_original, reduction='none')
    edge_score = edge_diff.view(edge_diff.size(0), -1).mean(dim=1)
    
    return edge_score

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


def detect_anomaly(model, data_loader, device, class_name=None, threshold=None):
    """
    Function to detect anomalies using both pixel-based and edge-based metrics.
    """

    model.eval()
    
    pixel_scores = []
    edge_scores = []
    all_data = []
    all_recon = []
    
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            data = data.to(device)
            
            reconstruction, _, _ = model(data)
            
            all_data.append(data.cpu())
            all_recon.append(reconstruction.cpu())
            
            # MAE
            pixel_error = torch.abs(reconstruction - data).view(data.size(0), -1).mean(dim=1)
            pixel_scores.extend(pixel_error.cpu().numpy())
            
            edge_error = compute_edge_score(data, reconstruction, class_name)
            edge_scores.extend(edge_error.cpu().numpy())
    
    pixel_scores = np.array(pixel_scores)
    edge_scores = np.array(edge_scores)
    
    # concatenate
    all_data = torch.cat(all_data, dim=0)
    all_recon = torch.cat(all_recon, dim=0)
    
    if threshold is None:
        threshold = np.percentile(pixel_scores, 95)
    
    # flaging anomalies
    pixel_anomalies = pixel_scores > threshold
    edge_anomalies = edge_scores > np.percentile(edge_scores, 95)
    
    return {
        'pixel_scores': pixel_scores,
        'edge_scores': edge_scores,
        'threshold': threshold,
        'pixel_anomalies': pixel_anomalies,
        'edge_anomalies': edge_anomalies,
        'data': all_data[:100], # limit because of visualization 
        'recon': all_recon[:100]
    }

def test_edge_computation():
    """Simple test to verify edge computation is working correctly."""
    # random test images
    test_image = torch.rand(2, 3, 64, 64)
    test_recon = torch.rand(2, 3, 64, 64)
    
    edge_score = compute_edge_score(test_image, test_recon)
    
    print(f"Edge score shape: {edge_score.shape}")
    print(f"Edge score values: {edge_score}")
    
    assert not torch.isnan(edge_score).any(), "Edge scores has NaN values"
    assert edge_score.shape[0] == test_image.shape[0], "Wrong number of scores returned"
    
    print("edge computation test passed")


if __name__ == "__main__":
    main()
