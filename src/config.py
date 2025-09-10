# Create a new file for configuration settings, hyperparameters, etc.

# src/config.py
import torch

# training parameters
RANDOM_SEED = 42
BATCH_SIZE = 16 # we can reduce to 8 or 4 if OOM on GPU
EPOCHS = 100
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-6
USE_SCHEDULER = True # ReduceLROnPlateau

# VAE parameters
LATENT_DIM = 128
BETA = 1.0 # KL weight, options: {0.5, 1.0, 2.0}
LAMBDA_GEO = 0.1 # Edge prior weight, options: {0.01, 0.1, 1.0}

# dataset parameters
IMAGE_SIZE = 128
CHANNELS = 3 # to handle grayscale by replicating to 3 channels
VALIDATION_SPLIT = 0.1 # 10% of train data for validation

# paths
DATA_PATH = '../mvtec_dataset.h5'
EXPERIMENT_PATH = './experiments'

# device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class-specific parameters
# will be populated dynamically based on dataset analysis
CLASS_STATS = {} # will contain mean, std per class
