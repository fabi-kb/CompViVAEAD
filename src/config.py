# config file structure is refined with LLM assistance and modified by Yavuz

"""
Central project configuration and small helpers.

Hyperparameters

Use compute_class_stats() to populate per-class
normalization statistics from the HDF5 dataset when available.

Note: This file does not perform heavy IO on import (compute_class_stats
runs only when called).
"""


import os
import json
import math
from typing import Dict, Tuple, Optional
import torch

# -----------------------------------------------------------------------------
# Randomness & device
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility"""
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic flags (may slow training) -  optional
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Training hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE_TRAIN = True

EPOCHS = 100
LEARNING_RATE = 1e-4
OPTIMIZER = "adam"
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-6

#scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "ReduceLROnPlateau" # or 'CosineAnnealingLR', 'StepLR'
SCHEDULER_PARAMS = {
    "mode": "min",
    "factor": 0.5,
    "patience": 5,
    "verbose": True,
    "min_lr": 1e-7
}

# save/eval per n epocjs
SAVE_EVERY = 5
EVAL_EVERY = 1
EARLY_STOPPING_PATIENCE = 15

# -----------------------------------------------------------------------------
# VAE / model parameters
# -----------------------------------------------------------------------------
IMAGE_SIZE = 128
CHANNELS = 3
LATENT_DIM = 128
BETA = 1.0 # KL weight {0.5,1.0,2.0}
LAMBDA_GEO = 0.1 # edge prior weight {0.01,0.1,1.0}

# edge prior options
EDGE_OPERATOR = "sobel" # can try 'canny' also (non-diff)
EDGE_ON_DENORM = True # compute edges on denormalized images in [0,1]
EDGE_GRAYSCALE = True

# -----------------------------------------------------------------------------
# Loss / normalization behavior (important: keep consistent across experiments)
# -----------------------------------------------------------------------------
# 'mean' reductions for image and edge MSE to get scale-invariant lambda_geo
RECON_LOSS_REDUCTION = "mean"
GEO_LOSS_REDUCTION = "mean"

# KL is computed as sum over latent dims; normalize by batch_size in training loop.

# -----------------------------------------------------------------------------
# Data paths & experiment paths
# -----------------------------------------------------------------------------
# Path to MVTec HDF5 dataset
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mvtec_dataset.h5"))

EXPERIMENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
# Per-run subfolders will be created under EXPERIMENT_ROOT/<class>/<run_id>/
MODEL_DIR = "models"
FIGS_DIR = "figs"
METRICS_DIR = "metrics"
CHECKPOINT_NAME = "model_epoch{epoch}.pt"
METRICS_JSON = "metrics.json"

# -----------------------------------------------------------------------------
# Normalization strategy: per_class or fallback to ImageNet-like stats
# -----------------------------------------------------------------------------
NORMALIZATION = "per_class" # 'per_class' - 'global' - 'imagenet_fallback'
# ImageNet fallback (only temporary; compute per-class means ASAP)
FALLBACK_MEAN = [0.485, 0.456, 0.406]
FALLBACK_STD  = [0.229, 0.224, 0.225]

# container for per-class mean/std (populated at runtime with compute_class_stats)
CLASS_STATS: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

# -----------------------------------------------------------------------------
# Misc / logging
# -----------------------------------------------------------------------------
LOG_INTERVAL = 50
TENSORBOARD = True
EXPERIMENT_INFO = {} # {'run_id':..., 'class':..., 'notes':...}

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def make_experiment_dirs(class_name: str, run_id: str) -> Dict[str, str]:
    """
    Create consistent directories for an experiment run and return absolute paths.
    structure: EXPERIMENT_ROOT/<class_name>/<run_id>/{models,figs,metrics}
    """
    base = os.path.join(EXPERIMENT_ROOT, class_name, run_id)
    paths = {
        "base": base,
        "models": os.path.join(base, MODEL_DIR),
        "figs": os.path.join(base, FIGS_DIR),
        "metrics": os.path.join(base, METRICS_DIR),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def validate_config(check_paths_exist: bool = False) -> None:
    """Sanity checks for config values. If check_paths_exist True, verifies DATA_PATH and EXPERIMENT_ROOT."""
    assert BATCH_SIZE > 0, "BATCH_SIZE must be > 0"
    assert 1 <= CHANNELS <= 4, "CHANNELS should be 1..4"
    assert IMAGE_SIZE > 0 and isinstance(IMAGE_SIZE, int)
    assert LATENT_DIM > 0
    assert 0.0 <= LAMBDA_GEO
    assert 0.0 <= BETA
    if check_paths_exist:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")
        os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

# -----------------------------------------------------------------------------
# example usage snippet (do NOT run on import)
# -----------------------------------------------------------------------------
# set_seed()
# validate_config(check_paths_exist=False)
# if NORMALIZATION == 'per_class':
#     compute_class_stats(max_samples=200)  # fill CLASS_STATS

# I put the function into utils.py (compute_class_stats())
