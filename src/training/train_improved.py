#  added parser fully with the help of an llm
#  description of funcs are refined/added

import os
import sys
import json
import time
import datetime
import argparse
from typing import Dict, List, Optional, Tuple, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from src.main_utils import build_parser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# from ..models.improved_vae import ImprovedVAE
# from ..data_loader import MVTecDataset
# from .losses import vae_geo_loss, sobel_edges
from src.models.improved_vae import ImprovedVAE
from src.data_loader import MVTecDataset
from src.training.losses import vae_geo_loss, sobel_edges
# from ..evaluation.visualization import visualize_reconstructions, visualize_edge_maps
from src.config import (
    RANDOM_SEED, DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    LEARNING_RATE, BETAS, WEIGHT_DECAY, OPTIMIZER,
    EPOCHS, BETA, LAMBDA_GEO,
    USE_SCHEDULER, SCHEDULER_TYPE, SCHEDULER_PARAMS,
    SAVE_EVERY, EVAL_EVERY, EARLY_STOPPING_PATIENCE,
    EXPERIMENT_ROOT, MODEL_DIR, FIGS_DIR, METRICS_DIR,
    CHECKPOINT_NAME, METRICS_JSON
)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    beta: float = BETA,
    lambda_geo: float = LAMBDA_GEO,
    num_epochs: int = EPOCHS,
    device: torch.device = DEVICE,
    experiment_dir: str = None,
    resume_from: str = None,
) -> Dict[str, List[float]]:
    """
    Train the improved VAE model with geometric prior loss.
    
    Args:
        model: ImprovedVAE model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        beta: Weight for KL divergence term
        lambda_geo: Weight for geometric prior term
        num_epochs: Number of training epochs
        device: Device to train on
        experiment_dir: Directory to save results
        resume_from: Checkpoint path to resume from
        
    Returns:
        Dictionary of training history
    """
    if experiment_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(EXPERIMENT_ROOT, f"run_{timestamp}")
    
    models_dir = os.path.join(experiment_dir, MODEL_DIR)
    figs_dir = os.path.join(experiment_dir, FIGS_DIR)
    metrics_dir = os.path.join(experiment_dir, METRICS_DIR)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon_loss': [], 'val_recon_loss': [],
        'train_kl_loss': [], 'val_kl_loss': [],
        'train_geo_loss': [], 'val_geo_loss': []
    }
    
    start_epoch = 0
    if resume_from is not None and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        print(f"Resuming from epoch {start_epoch}")
    
    # early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = {
            'loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0, 'geo_loss': 0.0
        }
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            images = batch.to(device)
            
            optimizer.zero_grad()
            recon_images, mu, logvar = model(images)
            
            loss_dict = vae_geo_loss(recon_images, images, mu, logvar, 
                                   beta=beta, lambda_geo=lambda_geo)
            
            loss = loss_dict['loss']
            
            loss.backward()
            optimizer.step()
            
            # update losses
            for key, value in loss_dict.items():
                train_losses[key] += value.item()
        
        # average train losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
            history[f'train_{key}'].append(train_losses[key])
        
        # validation
        model.eval()
        val_losses = {
            'loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0, 'geo_loss': 0.0
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")):
                images = batch.to(device)
                
                recon_images, mu, logvar = model(images)
                
                loss_dict = vae_geo_loss(recon_images, images, mu, logvar, 
                                       beta=beta, lambda_geo=lambda_geo)
                
                # update losses
                for key, value in loss_dict.items():
                    val_losses[key] += value.item()
        
        # average validaiton losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)
            history[f'val_{key}'].append(val_losses[key])
        
        # update learningrate scheduler
        if scheduler is not None:
            if SCHEDULER_TYPE == "ReduceLROnPlateau":
                scheduler.step(val_losses['loss'])
            else:
                scheduler.step()
        
        # epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_losses['loss']:.6f} (Recon: {train_losses['recon_loss']:.6f}, "
              f"KL: {train_losses['kl_loss']:.6f}, Geo: {train_losses['geo_loss']:.6f})")
        print(f"  Val Loss: {val_losses['loss']:.6f} (Recon: {val_losses['recon_loss']:.6f}, "
              f"KL: {val_losses['kl_loss']:.6f}, Geo: {val_losses['geo_loss']:.6f})")
        
        # save checkpoint for resuming ability
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(models_dir, CHECKPOINT_NAME.format(epoch=epoch+1))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': {
                    'beta': beta,
                    'lambda_geo': lambda_geo
                }
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            patience_counter = 0
            best_model_path = os.path.join(models_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': {
                    'beta': beta,
                    'lambda_geo': lambda_geo
                }
            }, best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
        
        # # generateing visualizations
        # if (epoch + 1) % EVAL_EVERY == 0 or epoch == num_epochs - 1:
        #     with torch.no_grad():
        #         sample_batch = next(iter(val_loader))
        #         sample_images = sample_batch.to(device)
        #         sample_recon, _, _ = model(sample_images)
                
        #         # reconstructions
        #         fig_recon = visualize_reconstructions(
        #             sample_images, sample_recon, 
        #             num_images=min(8, len(sample_images))
        #         )
        #         fig_recon.savefig(os.path.join(figs_dir, f"recon_epoch{epoch+1}.png"))
        #         plt.close(fig_recon)
                
        #         # edge maps
        #         fig_edges = visualize_edge_maps(
        #             sample_images, sample_recon, 
        #             edge_func=sobel_edges,
        #             num_images=min(4, len(sample_images))
        #         )
        #         fig_edges.savefig(os.path.join(figs_dir, f"edges_epoch{epoch+1}.png"))
        #         plt.close(fig_edges)
        
        with open(os.path.join(metrics_dir, METRICS_JSON), 'w') as f:
            json.dump(history, f, indent=4)
        
        # early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return history


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = OPTIMIZER,
    lr: float = LEARNING_RATE,
    betas: Tuple[float, float] = BETAS,
    weight_decay: float = WEIGHT_DECAY
) -> torch.optim.Optimizer:
    """Create optimizer based on config settings. we use adam by default."""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = SCHEDULER_TYPE,
    params: Dict = SCHEDULER_PARAMS
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create scheduler based on config settings. we use ReduceLROnPlateau by default."""
    if not USE_SCHEDULER:
        return None
        
    if scheduler_type == "ReduceLROnPlateau":
        # Filter out unsupported parameters
        valid_params = {}
        # List of valid parameters for ReduceLROnPlateau
        valid_keys = ['mode', 'factor', 'patience', 'threshold', 
                     'threshold_mode', 'cooldown', 'min_lr', 'eps']
        for key in valid_keys:
            if key in params:
                valid_params[key] = params[key]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **valid_params)
    # elif scheduler_type == "CosineAnnealingLR":
    #     return torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=params.get("T_max", 10), eta_min=params.get("eta_min", 0)
    #     )
    # elif scheduler_type == "StepLR":
    #     return torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=params.get("step_size", 10), gamma=params.get("gamma", 0.1)
    #     )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")


def main(
    class_name: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    data_subset: float = 1.0,
    val_split: float = 0.1,
    beta: float = BETA,
    lambda_geo: float = LAMBDA_GEO,
    run_id: Optional[str] = None,
    resume_from: Optional[str] = None
):
    """
    class_name: MVTec class to train on
    epochs: Number of training epochs
    batch_size: Batch size for training
    seed: Random seed
    data_subset: Fraction of training data to use
    val_split: Validation split ratio
    beta: Weight for KL divergence term
    lambda_geo: Weight for geometric prior term
    run_id: Custom run identifier
    resume_from: Checkpoint path to resume from
    """
    print(f"EXPERIMENT_ROOT = {EXPERIMENT_ROOT}")
    print(f"Will save to: {os.path.join(EXPERIMENT_ROOT, class_name, run_id)}")
    print(f"Working directory: {os.getcwd()}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # experiment directory
    if run_id is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{class_name}_improved_lambda{lambda_geo}_beta{beta}_seed{seed}"
    experiment_dir = os.path.join(EXPERIMENT_ROOT, class_name, run_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    config = {
        'class_name': class_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'seed': seed,
        'data_subset': data_subset,
        'val_split': val_split,
        'beta': beta,
        'lambda_geo': lambda_geo
    }
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Loading {class_name} dataset...")
    train_dataset = MVTecDataset(class_name=class_name, split='train')
    
    # subset data if requested
    if data_subset < 1.0:
        n_samples = int(data_subset * len(train_dataset))
        indices = torch.randperm(len(train_dataset))[:n_samples]
        train_dataset = Subset(train_dataset, indices)
    
    # split
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    print(f"Train samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
    
    # gett a sample to see input channels
    sample_batch = next(iter(train_loader))
    input_channels = sample_batch.shape[1]# [B, C, H, W]
    
    model = ImprovedVAE(input_channels=input_channels).to(DEVICE)
    print(f"Created ImprovedVAE model with {input_channels} input channels")
    
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    
    # train 
    print(f"Starting training with beta={beta}, lambda_geo={lambda_geo}")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        beta=beta,
        lambda_geo=lambda_geo,
        num_epochs=EPOCHS,
        device=DEVICE,
        experiment_dir=experiment_dir,
        resume_from=resume_from
    )
    
    final_metrics = {
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_recon_loss': history['train_recon_loss'][-1],
        'final_val_recon_loss': history['val_recon_loss'][-1],
        'final_train_kl_loss': history['train_kl_loss'][-1],
        'final_val_kl_loss': history['val_kl_loss'][-1],
        'final_train_geo_loss': history['train_geo_loss'][-1],
        'final_val_geo_loss': history['val_geo_loss'][-1],
        'best_val_loss': min(history['val_loss']),
        'epochs_trained': len(history['train_loss'])
    }
    
    with open(os.path.join(experiment_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print("Training complete!")
    print(f"Final validation loss: {final_metrics['final_val_loss']:.6f}")
    print(f"Best validation loss: {final_metrics['best_val_loss']:.6f}")
    print(f"Results saved to {experiment_dir}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    main(**vars(args))
