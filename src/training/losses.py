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
    compute Sobel edges from input tensor using differentiable convolutions.

    x: Input tensor of shape [B, C, H, W]

    retturns: edge map tensor of shape [B, 1, H, W]
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
    compute edge map loss between original and reconstructed images.

    x: original img [B, C, H, W]
    recon_x: reconstructed img [B, C, H, W]

    return: edge map loss (scalar tensor)
    """
    edges_x = sobel_edges(x)
    edges_recon = sobel_edges(recon_x)

    # MSE between edge maps with specified reduction
    loss = F.mse_loss(edges_recon, edges_x, reduction=GEO_LOSS_REDUCTION)# GEO_LOSS_REDUCTION = mean

    # # normalize by total elements
    # total_elements = edges_x.numel()
    # loss = loss / total_elements

    return loss


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict:
    """
    compute VAE loss components: reconstruction loss and KL divergence.

    x: original img [B, C, H, W]
    recon_x: reconstructed img [B, C, H, W]
    mu: mean of latent distribution of shape [B, latent_dim]
    logvar: log variance of latent distribution [B, latent_dim]
    beta: weight for KL divergence term
    reduction: reduction method for reconstruction loss ('mean' or 'sum')

    Returns:
        Dictionary containing total loss and individual components
    """

    batch_size = x.size(0)

    # reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    num_pixels = x.size(1) * x.size(2) * x.size(3)
    recon_loss = recon_loss * num_pixels # total per-sample loss
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # normalize
    # total_elements = x.size(1) * x.size(2) * x.size(3)
    kl_loss = kl_loss / batch_size
    # recon_loss = recon_loss / batch_size

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
    compute total VAE loss with geometric prior.

    x: Original img [B, C, H, W]
    recon_x: Reconstructed img [B, C, H, W]
    mu: Mean of latent distribution [B, latent_dim]
    logvar: Log variance of latent distribution [B, latent_dim]
    beta: weight for KL divergence term
    lambda_geo: weight for geometric loss term

    returns: dictionary of total loss and individual components
    """
    # vanilla VAE losses
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
