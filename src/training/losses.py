import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..config import EDGE_ON_DENORM, EDGE_GRAYSCALE, GEO_LOSS_REDUCTION


# For the geometric prior loss,:
# 1. Differentiable Sobel filter
# 2. Edge map computation function
# 3. Geometric loss function that compares edges between original and reconstructed images


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel edges from input tensor using differentiable convolutions.

    Args:
        x: Input tensor of shape [B, C, H, W]

    Returns:
        Edge map tensor of shape [B, 1, H, W]
    """
    # If x is grayscale with shape [B, 1, H, W], use as is
    # If x is RGB [B, 3, H, W] and EDGE_GRAYSCALE=True, convert to grayscale
    if x.shape[1] == 3 and EDGE_GRAYSCALE:
        # RGB to grayscale conversion weights (luminance)
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(
            1, 3, 1, 1
        )
        x = torch.sum(x * rgb_weights, dim=1, keepdim=True)

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)

    # padding
    pad = nn.ReplicationPad2d(1)
    x_padded = pad(x)

    # convolution for each channel indpendently if not already grayscale
    if x.shape[1] == 1:
        g_x = F.conv2d(x_padded, sobel_x)
        g_y = F.conv2d(x_padded, sobel_y)
    else:
        b, c, h, w = x_padded.shape
        x_reshaped = x_padded.reshape(b * c, 1, h, w)
        g_x = F.conv2d(x_reshaped, sobel_x).reshape(b, c, h - 2, w - 2)
        g_y = F.conv2d(x_reshaped, sobel_y).reshape(b, c, h - 2, w - 2)

    # edge magnitude (L2 norm)
    edges = torch.sqrt(g_x**2 + g_y**2 + 1e-8)

    # if multi-channel, take mean
    if edges.shape[1] > 1:
        edges = torch.mean(edges, dim=1, keepdim=True)

    return edges


def edge_map_loss(x: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
    """
    Compute edge map loss between original and reconstructed images.

    Args:
        x: Original images tensor of shape [B, C, H, W]
        recon_x: Reconstructed images tensor of shape [B, C, H, W]

    Returns:
        Edge map loss (scalar tensor)
    """
    edges_x = sobel_edges(x)
    edges_recon = sobel_edges(recon_x)

    # MSE between edge maps with specified reduction
    loss = F.mse_loss(edges_recon, edges_x, reduction=GEO_LOSS_REDUCTION)

    return loss


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict:
    """
    Compute VAE loss components: reconstruction loss and KL divergence.

    Args:
        recon_x: Reconstructed images tensor of shape [B, C, H, W]
        x: Original images tensor of shape [B, C, H, W]
        mu: Mean of latent distribution of shape [B, latent_dim]
        logvar: Log variance of latent distribution of shape [B, latent_dim]
        beta: Weight for KL divergence term
        reduction: Reduction method for reconstruction loss ('mean' or 'sum')

    Returns:
        Dictionary containing total loss and individual components
    """

    batch_size = x.size(0)
    # reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    num_pixels = x.size(1) * x.size(2) * x.size(3)
    recon_loss = recon_loss * num_pixels  # total per-sample loss

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # normalize
    kl_loss = kl_loss / batch_size
    recon_loss = recon_loss / batch_size

    total_loss = recon_loss + beta * kl_loss

    return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}


def vae_geo_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    lambda_geo: float = 0.1,
) -> dict:
    """
    Compute total VAE loss with geometric prior.

    Args:
        recon_x: Reconstructed images tensor of shape [B, C, H, W]
        x: Original images tensor of shape [B, C, H, W]
        mu: Mean of latent distribution of shape [B, latent_dim]
        logvar: Log variance of latent distribution of shape [B, latent_dim]
        beta: Weight for KL divergence term
        lambda_geo: Weight for geometric loss term

    Returns:
        Dictionary containing total loss and individual components
    """
    # standard VAE losses
    vae_losses = vae_loss(recon_x, x, mu, logvar, beta)

    # geometric prior loss
    geo_loss = edge_map_loss(x, recon_x)

    # total loss with geometric prior
    total_loss = vae_losses["loss"] + lambda_geo * geo_loss

    return {
        "loss": total_loss,
        "recon_loss": vae_losses["recon_loss"],
        "kl_loss": vae_losses["kl_loss"],
        "geo_loss": geo_loss,
    }
