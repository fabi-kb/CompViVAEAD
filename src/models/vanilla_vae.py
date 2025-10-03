import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        input_channels=3,
        latent_dim=128,
        hidden_layers=[32, 64, 128, 256, 512],
        image_size=128,
    ):
        super(VAE, self).__init__()

        self.config = {
            'input_channels': input_channels,
            'latent_dim': latent_dim,
            'hidden_layers': hidden_layers,
            'image_size': image_size
        }

        self._latent_dim = latent_dim
        self._input_channels = input_channels

        self._hidden_layers = hidden_layers

        encoder_layers = []
        in_channels = input_channels

        for h in hidden_layers:
            encoder_layers.extend(
                [
                    nn.Conv2d(in_channels, h, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h),
                    nn.LeakyReLU(),
                ]
            )
            in_channels = h

        self.encoder = nn.Sequential(*encoder_layers)

        # achieve flattened size
        self.spatial_dim = image_size // (2 ** len(hidden_layers))

        self.flattened_size = in_channels * self.spatial_dim * self.spatial_dim

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

        # decoder

        decoder_layers = []
        dec_hidden_layers = list(reversed(hidden_layers))

        in_channels = dec_hidden_layers[0]

        for h in dec_hidden_layers:
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        h,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(h),
                    nn.LeakyReLU(),
                ]
            )
            in_channels = h

        decoder_layers.extend(
            [
                nn.Conv2d(
                    in_channels, out_channels=input_channels, kernel_size=3, padding=1
                ),
                nn.Sigmoid(),
            ]
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x):
        x = self.fc_decode(x)
        x = x.view(-1, self._hidden_layers[-1], self.spatial_dim, self.spatial_dim)

        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)

        return self.decode(z), mu, logvar
