import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

from ..config import LATENT_DIM, IMAGE_SIZE, CHANNELS


class ImprovedVAE(nn.Module):
    """
    Improved VAE with geometric prior loss.
    input_channels: Number of input image channels (3 for RGB, 1 for grayscale)
    """
    
    def __init__(
        self,
        input_channels=3,
        latent_dim=128,
        hidden_dims=[32, 64, 128, 256, 512],
        image_size=128,
    ):
        super(ImprovedVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        
        # encoder
        modules = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # feature map size
        self.feature_size = IMAGE_SIZE // (2 ** len(hidden_dims))
        self.feature_volume = hidden_dims[-1] * self.feature_size * self.feature_size
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.feature_volume, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_volume, latent_dim)
        
        # decoder
        modules = []
        
        # initial projection from latent space
        self.decoder_input = nn.Linear(latent_dim, self.feature_volume)        
        hidden_dims.reverse()
        
        # transposed convolutions
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        # final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    hidden_dims[-1], input_channels, kernel_size=3, padding=1
                ),
                nn.Sigmoid()
            )
        )
        
        self.decoder = nn.Sequential(*modules)        
        hidden_dims.reverse()
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Input tensor of shape [B, C, H, W]
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        
        # latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.

        Returns: z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decoder_input(z)
        
        # reshaping to feature maps
        z = z.view(-1, self.hidden_dims[-1], self.feature_size, self.feature_size)
        
        recon_x = self.decoder(z)
        
        return recon_x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        
        # sample latent vector
        z = self.reparameterize(mu, logvar)
        
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar
    
    def sample(
        self,
        num_samples: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        sampleing from the latent space.
            
        reutrned samples: Generated samples of shape [num_samples, C, H, W]
        """
        if device is None:
            device = next(self.parameters()).device
            
        # sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        samples = self.decode(z)
        
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        reconstruct input images.

        x: Input tensor of shape [B, C, H, W]
            
        Returns: recon_x: Reconstructed images of shape [B, C, H, W]
        """
        # without sampling (use mean)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x
