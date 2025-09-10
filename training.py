import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def train_vae(
    model, train_loader, val_loader, epochs=50, lr=1e-3, beta=1.0, device="cpu"
):
    """
    Implements the training loop for the VAE, uses the adam optimizer

    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()

        train_loss = 0

        for idx, data in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()

            reconstruction, mu, logvar = model(data)

            loss, recon_loss, kl_loss = vae_loss(reconstruction, data, mu, logvar, beta)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for _, data in enumerate(val_loader):
                data = data.to(device)

                reconstruction, mu, logvar = model(data)

                loss, _, _ = vae_loss(reconstruction, data, mu, logvar, beta)

                val_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n Recon Loss: {recon_loss.item()/len(data):.4f}, KL Loss: {kl_loss.item()/len(data):.4f}"
        )


def vae_loss(reconstruction, original, mu, logvar, beta):

    recon_loss = F.mse_loss(reconstruction, original, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def detect_anomaly(model, data_loader, device, threshold=None):
    """
    Function to detect anomalies checks the reconstruction error and if its larger then sets the mask to true
    """

    model.eval()

    recons_errors = []

    with torch.no_grad():
        for _, data in enumerate(data_loader):
            data = data.to(device)

            reconstruction, _, _ = model(data)

            errors = F.mse_loss(reconstruction, data, reduction="none")
            errors = errors.view(errors.size(0), -1).mean(dim=1)
            recons_errors.extend(errors.cpu().numpy())

    recons_errors = np.array(recons_errors)

    if threshold is None:
        threshold = np.percentile(recons_errors, 95)

    anomalies = recons_errors > threshold

    return recons_errors, anomalies, threshold
