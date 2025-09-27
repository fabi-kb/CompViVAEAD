# example script command to run:
# python src/training/train_baseline.py --class_name metal_nut --epochs 100 --batch_size 16 --seed 42 --beta 0.05 --run_id metal_nut_baseline_beta0.05_seed42
# python src/training/train_baseline.py --class_name bottle --epochs 100 --batch_size 64 --seed 42 --beta 0.05 --run_id bottle_baseline_beta0.05_seed42

import os
import sys
import json
import torch
import torch.nn as nn
import datetime
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
import torchvision

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.vanilla_vae import VAE
from src.config import *
from src.data_loader import H5Dataset, MVTecDataset
from src.main_utils import build_parser
from src.training.losses import vae_loss

# from src.evaluation.evaluate import visualize_reconstructions

from datetime import datetime


def train_vae(
    model,
    train_loader,
    val_loader,
    device,
    config,
    experiment_dir=None,
    save_every=5,
):
    """
    implements the training loop for the VAE, uses the adam optimizer

    model: VAE
    train_loader: dataLoader for training data
    val_loader: dataLoader for validation data

    epochs
    lr: learning rate
    beta: weight for KL divergence term

    device: device to train on
    experiment_dir: directory to save results
    save_every: save checkpoint every N epochs
    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    metrics = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
    }

    best_val_loss = float("inf")

    # models directory
    if experiment_dir:
        models_dir = os.path.join(experiment_dir, "models")
        figs_dir = os.path.join(experiment_dir, "figs")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)

    for epoch in range(config['epochs']):
        model.train()

        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            losses = vae_loss(recon, data, mu, logvar, beta=config['beta'])
            loss = losses["loss"]
            recon_loss = losses["recon_loss"]
            kl_loss = losses["kl_loss"]

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
        avg_loss = train_loss / len(train_loader)
        avg_recon_loss = train_recon_loss / len(train_loader)
        avg_kl_loss = train_kl_loss / len(train_loader)

        model.eval()

        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0

        with torch.no_grad():

            for _, data in enumerate(val_loader):
                data = data.to(device)
                recon, mu, logvar = model(data)
                losses = vae_loss(recon, data, mu, logvar, beta=config['beta'])
                val_loss += losses["loss"].item()
                val_recon_loss += losses["recon_loss"].item()
                val_kl_loss += losses["kl_loss"].item()

            # visualize some reconstructions
            if experiment_dir:
                fig_path = os.path.join(figs_dir, f"recon_epoch{epoch+1}.png")
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon[:n]])
                comparison = comparison.cpu()
                img_grid = torchvision.utils.make_grid(comparison, nrow=n)

                plt.figure(figsize=(12, 6))
                plt.axis("off")
                plt.title(f"Epoch {epoch+1} Reconstructions")
                plt.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
                plt.savefig(fig_path)
                plt.close()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_kl_loss = val_kl_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        scheduler.step(val_loss)

        metrics["epochs"].append(epoch + 1)
        metrics["train_loss"].append(avg_loss)
        metrics["val_loss"].append(avg_val_loss)
        metrics["train_recon_loss"].append(avg_recon_loss)
        metrics["train_kl_loss"].append(avg_kl_loss)
        metrics["val_recon_loss"].append(avg_val_recon_loss)
        metrics["val_kl_loss"].append(avg_val_kl_loss)

        print(
            f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
            f"Train Recon: {train_recon_loss:.4f}, Train KL: {train_kl_loss:.4f}\n"
            f"Val Recon: {val_recon_loss:.4f}, Val KL: {val_kl_loss:.4f}"
        )

        if experiment_dir:
            with open(os.path.join(experiment_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

        # # checkpoint
        # if experiment_dir and ((epoch + 1) % save_every == 0 or epoch + 1 == epochs):
        #     checkpoint_path = os.path.join(models_dir, f"model_epoch{epoch+1}.pt")
        #     torch.save(
        #         {
        #             "epoch": epoch + 1,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "scheduler_state_dict": scheduler.state_dict(),
        #             "metrics": metrics,
        #         },
        #         checkpoint_path,
        #     )
        #     print(f"Saved checkpoint to {checkpoint_path}")

        # best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if experiment_dir:
                best_model_path = os.path.join(models_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "metrics": metrics,
                        "val_loss": val_loss,
                    },
                    best_model_path,
                )
                print(f"Saved best model with val_loss: {val_loss:.4f}")

    return model, metrics


def main(
    class_name,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    data_subset: float = 1.0,
    val_split: float = 0.1,
    beta: float = BETA,
    learning_rate: float = LEARNING_RATE,
    run_id: Optional[str] = None,
    device: str = DEVICE,
    **kwargs,
):
    """
    Args:
    class_name: MVTec class name
    epochs
    batch_size
    seed: random seed
    beta: weight for KL divergence term
    run_id: experiment id

    data_subset: fraction of training data to use (0-1)
    val_split: fraction of training data to use for validation (0-1)
    """

    run_config = {
        # identification
        'class_name': class_name,
        'run_id': run_id if run_id else f"{class_name}_baseline_beta{beta}_seed{seed}",
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        
        # training parameters
        'epochs': epochs,
        'batch_size': batch_size,
        'seed': seed,
        'data_subset': data_subset,
        'val_split': val_split,
        'learning_rate': learning_rate,
        
        # model hyperparameters
        'beta': beta,
        'latent_dim': LATENT_DIM,
        'image_size': IMAGE_SIZE,
        
        # architecture
        'input_channels': 3,
        'hidden_layers': [32, 64, 128, 256, 512],
        
        # optimizer settings
        'optimizer': 'Adam',
        'weight_decay': 0.0,
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
    }
    
    torch.manual_seed(run_config['seed'])
    np.random.seed(run_config['seed'])

    print(f"Using device: {device}")

    experiment_dir = os.path.join(EXPERIMENT_ROOT, run_config['class_name'], "baseline", run_config['run_id'])
    os.makedirs(experiment_dir, exist_ok=True)
    print("ACTIVE CONFIGURATION:")
    for key, value in run_config.items():
        print(f"  {key}: {value}")
        
    print(run_config)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    num_workers = min(8, os.cpu_count() // 2)

    print(f"Loading {class_name} dataset...")
    train_dataset = MVTecDataset(class_name=run_config['class_name'], split="train")

    # applying data subset if specified
    if data_subset < 1.0:
        subset_size = int(len(train_dataset) * data_subset)
        indices = torch.randperm(len(train_dataset))[:subset_size]

        train_dataset = Subset(train_dataset, indices)

    # split
    dataset_size = len(train_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(run_config['seed']),
    )

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    train_loader = DataLoader(
        train_subset,
        batch_size=run_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=run_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    model = VAE(
        input_channels=run_config['input_channels'],
        latent_dim=run_config['latent_dim'],
        hidden_layers=run_config['hidden_layers'],
        image_size=run_config['image_size']
    ).to(device)

    print(
        f"Created VAE model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    print(f"Model architecture:\n{model}")

    # training
    model, metrics = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=run_config,
        experiment_dir=experiment_dir,
        save_every=5
    )

    final_metrics = {
        'final_train_loss': metrics['train_loss'][-1],
        'final_val_loss': metrics['val_loss'][-1],
        'best_val_loss': min(metrics['val_loss']),
        'config': run_config
    }
    
    with open(os.path.join(experiment_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"Training complete. Results saved to {experiment_dir}")


if __name__ == "__main__":
    # if build_parser does not have all required args, we can extend it
    parser = build_parser()
    args = parser.parse_args()

    main(**vars(args))
