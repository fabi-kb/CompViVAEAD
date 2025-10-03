# example script command to run:
# python src/evaluation/evaluate.py --exp_dir experiments/carpet/improved/carpet_improved_lambda0.01_beta1.0_seed42

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.vanilla_vae import VAE
from src.models.improved_vae import ImprovedVAE
from src.data_loader import MVTecDataset
from src.config import DEVICE, EDGE_ON_DENORM, EDGE_GRAYSCALE, CLASS_STATS
from src.training.losses import sobel_edges


def load_model(exp_dir: str) -> Tuple[torch.nn.Module, str]:
    """
    load the best model
            
    returns (loaded model, class name)
    """
    best_model_path = os.path.join(exp_dir, "models", "best_model.pt")

    if not os.path.exists(best_model_path):
        # Try latest checkpoint
        models_dir = os.path.join(exp_dir, "models")
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"No models directory found at {models_dir}")
            
        model_files = [f for f in os.listdir(models_dir) 
                      if f.startswith("model_epoch") and f.endswith(".pt")]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        best_model_path = os.path.join(models_dir, sorted(model_files)[-1])
    
    print(f"Loading model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    
    # Load config to determine model type and parameters
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = checkpoint.get("config", {})
    
    class_name = config.get("class_name", os.path.basename(os.path.dirname(exp_dir)))
    print(f"Class name: {class_name}")
    
    is_improved = "lambda_geo" in config
    
    # create appropriate model
    input_channels = 3
    if is_improved:
        print(f"Creating ImprovedVAE with lambda_geo={config.get('lambda_geo', 'unknown')}")
        
        hidden_dims = config.get('hidden_dims', [32, 64, 128, 256, 512])
        latent_dim = config.get('latent_dim', 128)
        
        model = ImprovedVAE(
            input_channels=input_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )
    else:
        print("Creating baseline VAE")
        
        # get parameters from config
        hidden_layers = config.get('hidden_layers', [32, 64, 128, 256, 512])
        latent_dim = config.get('latent_dim', 128)
        
        model = VAE(
            input_channels=input_channels, 
            latent_dim=latent_dim,
            hidden_layers=hidden_layers
        )
    
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        print(f"Warning: Error loading model with strict matching: {e}")
        print("Trying to load with strict=False...")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Model loaded with missing or unexpected keys (architecture mismatch)")
    
    model = model.to(DEVICE)
    model.eval()
    
    return model, class_name


def denormalize_image(img: torch.Tensor, class_name: str) -> torch.Tensor:
    """
    Denormalize image tensor based on class statistics.
    
    img: Input image tensor [B, C, H, W]
    class_name: Class name for statistics lookup
        
    returns denormalized image tensor
    """
    if not EDGE_ON_DENORM:
        return img
        
    # class stats if available
    if class_name in CLASS_STATS and "mean" in CLASS_STATS[class_name]:
        mean = torch.tensor(CLASS_STATS[class_name]["mean"]).view(1, -1, 1, 1).to(img.device)
        std = torch.tensor(CLASS_STATS[class_name]["std"]).view(1, -1, 1, 1).to(img.device)
    else:
        # Fallback stats (ImageNet) default
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(img.device)
    
    # denormalize: img * std + mean
    return img * std + mean


def compute_anomaly_scores(
    model: torch.nn.Module, 
    dataloader: DataLoader,
    class_name: str,
    return_images: bool = False,
    max_images: int = 100
) -> Dict[str, np.ndarray]:
    """
    compute anomaly scores (pixel MAE and edge-based) for a dataset
    
    model: VAE model
    return_images: whether to return original and reconstructed images
    max_images: max number of images to return

    returns: dictionary with anomaly scores and optionally images
    """
    model.eval()
    
    pixel_scores = []
    edge_scores = []
    originals = []
    reconstructions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing anomaly scores"):
            # MVTecDataset returns images directly
            images = batch.to(DEVICE)

            if len(images.shape) > 4: # if extra dimension
                images = images.squeeze(1) #remove
            
            # reconstructions
            recons, _, _ = model(images)
            
            # primary score: pixel-wise MAE (score_MAE)
            pixel_mae = torch.mean(torch.abs(images - recons), dim=[1, 2, 3])
            pixel_scores.append(pixel_mae.cpu().numpy())

            # secondary score: edge-based score (score_edge)
            # handle denormalization if required
            if EDGE_ON_DENORM:
                images_for_edge = denormalize_image(images, class_name)
                recons_for_edge = denormalize_image(recons, class_name)
            else:
                images_for_edge = images
                recons_for_edge = recons
            
            # grayscale conversion for edge detection
            if EDGE_GRAYSCALE and images.shape[1] == 3:
                # RGB to grayscale
                images_gray = 0.299 * images_for_edge[:,0:1] + 0.587 * images_for_edge[:,1:2] + 0.114 * images_for_edge[:,2:3]
                recons_gray = 0.299 * recons_for_edge[:,0:1] + 0.587 * recons_for_edge[:,1:2] + 0.114 * recons_for_edge[:,2:3]
                
                edge_orig = sobel_edges(images_gray)
                edge_recon = sobel_edges(recons_gray)
            else:
                edge_orig = sobel_edges(images_for_edge)
                edge_recon = sobel_edges(recons_for_edge)
            
            # edge-based score
            edge_score = torch.mean(torch.abs(edge_orig - edge_recon), dim=[1, 2, 3])
            edge_scores.append(edge_score.cpu().numpy())
            
            # save if requested
            if return_images and len(originals) * images.size(0) < max_images:
                originals.append(images.cpu())
                reconstructions.append(recons.cpu())
    
    pixel_scores = np.concatenate(pixel_scores)
    edge_scores = np.concatenate(edge_scores)
    
    result = {
        'pixel_scores': pixel_scores,
        'edge_scores': edge_scores,
    }
    
    # add images if requested
    if return_images:
        if originals:
            result['original_images'] = torch.cat(originals, dim=0)[:max_images]
            result['reconstructed_images'] = torch.cat(reconstructions, dim=0)[:max_images]
    
    return result


def evaluate_dataset(
    model: torch.nn.Module, 
    class_name: str,
    dataset_type: str = 'test',
    defect_type: Optional[str] = None,
    is_anomaly: bool = True,
    output_dir: Optional[str] = None,
    batch_size: int = 16,
    return_images: bool = False
) -> Dict[str, Any]:
    """
    evaluate a model on a specific dataset
    
    VAE model
    class_name
    dataset_type: train, test, synthetic
    defect_type: (cutout, scratches)
    is_anomaly: if the dataset contains anomalies
    output_dir: dir for saving results
    batch_size
    return_images: if to return images for visualization

    returns: dictionary with evaluation results
    """
    dataset = MVTecDataset(class_name=class_name, split=dataset_type, defect_type=defect_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Evaluating {class_name} - {dataset_type}" + (f" ({defect_type})" if defect_type else ""))
    print(f"Dataset size: {len(dataset)}")
    
    # anomaly scores
    results = compute_anomaly_scores(model, dataloader, class_name, return_images)
    
    # ground truth labels
    if is_anomaly:
        labels = np.ones_like(results['pixel_scores'])
    else:
        labels = np.zeros_like(results['pixel_scores'])
    
    results['labels'] = labels
    
    # save per-image results if output_dir
    if output_dir:
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        per_image_scores = {
            'image_ids': [f"img_{i}" for i in range(len(results['pixel_scores']))], # add image IDs
            'pixel_scores': results['pixel_scores'].tolist(),
            'edge_scores': results['edge_scores'].tolist(),
            'labels': labels.tolist(),
            'dataset_type': dataset_type,
            'defect_type': defect_type
        }
        
        scores_file = f"per_image_scores_{dataset_type}"
        if defect_type:
            scores_file += f"_{defect_type}"
        scores_file += ".json"
        
        with open(os.path.join(metrics_dir, scores_file), 'w') as f:
            json.dump(per_image_scores, f, indent=2)
        
        # for easier loading (efficient)
        scores_np_file = scores_file.replace('.json', '.npz')
        np.savez(
            os.path.join(metrics_dir, scores_np_file),
            pixel_scores=results['pixel_scores'],
            edge_scores=results['edge_scores'],
            labels=labels
        )
    
    return results


def compute_evaluation_metrics(
    scores: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    compute evaluation metrics for anomaly detection
    
    scores: anomaly scores
    labels: ground truth labels (0=normal, 1=anomaly)

    returns dictionary of evaluation metrics
    """
    # ROC curve and AUC
    fpr, tpr, thresholds_roc = metrics.roc_curve(labels, scores)
    roc_auc = metrics.roc_auc_score(labels, scores)
    
    # Precision-Recall curve and AP
    precision, recall, thresholds_pr = metrics.precision_recall_curve(labels, scores)
    ap = metrics.average_precision_score(labels, scores)
    
    # Precision at 5% FPR
    idx_at_5pct = np.argmax(fpr >= 0.05)
    precision_at_5pct = precision[idx_at_5pct]
    
    # F1 score for optimal threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else thresholds_pr[-1]
    max_f1 = f1_scores[optimal_idx]
    
    return {
        "roc_auc": float(roc_auc),
        "ap": float(ap),
        "precision_at_5pct": float(precision_at_5pct),
        "max_f1": float(max_f1),
        "optimal_threshold": float(optimal_threshold),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds_roc": thresholds_roc.tolist(),
        "thresholds_pr": thresholds_pr.tolist()
    }


def save_roc_pr_curves(metrics_dict: Dict, output_dir: str, score_type: str, class_name: str):
    """
    Save ROC and PR curves as PNG files.
    
    metrics_dict: dictionary of evaluation metrics
    output_dir
    score_type: ('pixel' or 'edge')
    class_name, for plot title
    """
    figs_dir = os.path.join(output_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_dict["fpr"], metrics_dict["tpr"], 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.grid(True)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{class_name} - ROC Curve ({score_type})\nAUC: {metrics_dict["roc_auc"]:.4f}')
    plt.savefig(os.path.join(figs_dir, f"roc_{score_type}.png"), dpi=300)
    plt.close()
    
    # PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_dict["recall"], metrics_dict["precision"], 'r-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{class_name} - Precision-Recall Curve ({score_type})\nAP: {metrics_dict["ap"]:.4f}')
    plt.savefig(os.path.join(figs_dir, f"pr_{score_type}.png"), dpi=300)
    plt.close()


def create_reconstruction_gallery(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    output_dir: str,
    class_name: str,
    selection_indices: Optional[List[int]] = None,
    max_samples: int = 10
):
    """
    create a gallery of reconstructions with original/recon/diff/edges.
    
    original_images: tensor[N, C, H, W]
    reconstructed_images:  tensor[N, C, H, W]
    output_dir
    class_name, for gallery title
    selection_indices: specific indices to select for gallery
    max_samples, to include in gallery
    """
    figs_dir = os.path.join(output_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    # select images for gallery
    if selection_indices is not None:
        selection_indices = selection_indices[:max_samples]
        originals = original_images[selection_indices]
        recons = reconstructed_images[selection_indices]
    else:
        # or first N samples
        originals = original_images[:max_samples]
        recons = reconstructed_images[:max_samples]
    
    num_samples = len(originals)
    
    # gallery with 5 columns (orig/recon/diff/edge_orig/edge_recon)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # 2 rows x 5 columns 
    for i in range(min(2, num_samples)):
        orig = originals[i].permute(1, 2, 0).cpu().numpy()
        orig = np.clip(orig, 0, 1)
        
        recon = recons[i].permute(1, 2, 0).cpu().numpy()
        recon = np.clip(recon, 0, 1)
        
        # absolute difference
        diff = np.abs(orig - recon)
        diff = diff / diff.max() if diff.max() > 0 else diff
        
        # edge maps
        orig_tensor = originals[i].unsqueeze(0)  # Add batch dimension
        recon_tensor = recons[i].unsqueeze(0)
        
        # converting to grayscale for edge detection
        if EDGE_GRAYSCALE and orig_tensor.shape[1] == 3:
            orig_gray = 0.299 * orig_tensor[:,0:1] + 0.587 * orig_tensor[:,1:2] + 0.114 * orig_tensor[:,2:3]
            recon_gray = 0.299 * recon_tensor[:,0:1] + 0.587 * recon_tensor[:,1:2] + 0.114 * recon_tensor[:,2:3]
        else:
            orig_gray = orig_tensor
            recon_gray = recon_tensor
        
        edge_orig = sobel_edges(orig_gray)[0].squeeze().cpu().numpy()
        edge_recon = sobel_edges(recon_gray)[0].squeeze().cpu().numpy()
        
        # original
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        # reconstruction
        axes[i, 1].imshow(recon)
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis('off')
        
        # difference
        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title("Abs. Difference")
        axes[i, 2].axis('off')

        # edge maps
        axes[i, 3].imshow(edge_orig, cmap='viridis')
        axes[i, 3].set_title("Original Edges")
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(edge_recon, cmap='viridis')
        axes[i, 4].set_title("Recon Edges")
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Reconstruction Gallery - {class_name}", y=1.02)
    plt.savefig(os.path.join(figs_dir, f"gallery_{class_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()


    # saving individual images for visualization step for later
    examples_dir = os.path.join(output_dir, "recon_examples")
    os.makedirs(examples_dir, exist_ok=True)

    for i in range(min(len(original_images), max_samples)):
        # original image
        orig_img = original_images[i].permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)
        Image.fromarray(orig_img).save(os.path.join(examples_dir, f"img_{i}_orig.png"))
        
        # reconstructed image
        recon_img = reconstructed_images[i].permute(1, 2, 0).cpu().numpy()
        recon_img = np.clip(recon_img * 255, 0, 255).astype(np.uint8)
        Image.fromarray(recon_img).save(os.path.join(examples_dir, f"img_{i}_recon.png"))
        
        # difference image
        diff = np.abs(orig_img - recon_img)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255
        diff = diff.astype(np.uint8)
        Image.fromarray(diff).save(os.path.join(examples_dir, f"img_{i}_diff.png"))
        
        # edge maps
        edge_orig = sobel_edges(original_images[i:i+1])[0].cpu().numpy()

        #we got an error here with single channel images
        # PIL cannot handle (H, W, 1) images directly, so we add a check
        if edge_orig.shape[0] == 1: # single channel (1, H, W)
            edge_orig = edge_orig[0] # Convert to (H, W)
        else: # multi-channel - convert to RGB
            edge_orig = np.transpose(edge_orig, (1, 2, 0))
            if edge_orig.shape[2] == 1:
                edge_orig = edge_orig[:, :, 0]

        edge_orig = np.clip(edge_orig * 255, 0, 255).astype(np.uint8)
        Image.fromarray(edge_orig).save(os.path.join(examples_dir, f"img_{i}_edge_orig.png"))

        # reconstruction edges
        edge_recon = sobel_edges(reconstructed_images[i:i+1])[0].cpu().numpy()

        #same check for single/multi-channel
        if edge_recon.shape[0] == 1:
            edge_recon = edge_recon[0]
        else:
            edge_recon = np.transpose(edge_recon, (1, 2, 0))
            if edge_recon.shape[2] == 1:
                edge_recon = edge_recon[:, :, 0]

        edge_recon = np.clip(edge_recon * 255, 0, 255).astype(np.uint8)
        Image.fromarray(edge_recon).save(os.path.join(examples_dir, f"img_{i}_edge_recon.png"))


def evaluate_and_save_results(
    model: torch.nn.Module, 
    class_name: str, 
    output_dir: str,
    batch_size: int = 16
):
    """
    evaluate model on both test data and synthetic anomalies
    save all results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_summary = {
        "class_name": class_name,
        "test_sets": [],
        "synthetic_sets": []
    }
    
    #Evaluate
    # 1. normal training data (for calibration)
    normal_results = evaluate_dataset(
        model, class_name, 
        dataset_type='train', 
        defect_type='good', 
        is_anomaly=False,
        output_dir=output_dir,
        batch_size=batch_size,
        return_images=True
    )
    
    # 2. real test data with defects
    test_results = evaluate_dataset(
        model, class_name, 
        dataset_type='test', 
        is_anomaly=True,
        output_dir=output_dir,
        batch_size=batch_size,
        return_images=True
    )
    
    # 3. synthetic cutout defects
    try:
        cutout_results = evaluate_dataset(
            model, class_name, 
            dataset_type='cutout_synth', 
            is_anomaly=True,
            output_dir=output_dir,
            batch_size=batch_size,
            return_images=True
        )
    except Exception as e:
        print(f"Cutout synthetic dataset not found. We Skipp... {e}")
        cutout_results = None

    # 4. synthetic scratch defects
    try:
        scratch_results = evaluate_dataset(
            model, class_name, 
            dataset_type='scratches_synth', 
            is_anomaly=True,
            output_dir=output_dir,
            batch_size=batch_size,
            return_images=True
        )
    except Exception as e:
        print(f"Scratches synthetic dataset not found. We Skipp... {e}")
        scratch_results = None
    
    # for real test data
    # we combine normal (train/good) and anomaly (test) data
    combined_pixel_scores = np.concatenate([normal_results['pixel_scores'], test_results['pixel_scores']])
    combined_edge_scores = np.concatenate([normal_results['edge_scores'], test_results['edge_scores']])
    combined_labels = np.concatenate([normal_results['labels'], test_results['labels']])
    
    # pixel MAE score
    pixel_metrics = compute_evaluation_metrics(combined_pixel_scores, combined_labels)
    
    # edge score
    edge_metrics = compute_evaluation_metrics(combined_edge_scores, combined_labels)
    
    metrics_summary["test_sets"].append({
        "dataset_type": "real_test",
        "num_normal": len(normal_results['pixel_scores']),
        "num_anomaly": len(test_results['pixel_scores']),
        "pixel_metrics": pixel_metrics,
        "edge_metrics": edge_metrics
    })
    
    save_roc_pr_curves(pixel_metrics, output_dir, "pixel", class_name)
    save_roc_pr_curves(edge_metrics, output_dir, "edge", class_name)
    
    # for normal samples
    create_reconstruction_gallery(
        normal_results['original_images'][:5], 
        normal_results['reconstructed_images'][:5],
        output_dir, 
        f"{class_name}_normal"
    )
    
    # for anomaly samples
    create_reconstruction_gallery(
        test_results['original_images'][:5], 
        test_results['reconstructed_images'][:5],
        output_dir, 
        f"{class_name}_anomaly"
    )
    
    # evaluate synthetic data if exists
    if cutout_results:
        # we combine normal and cutout data
        combined_pixel_scores = np.concatenate([normal_results['pixel_scores'], cutout_results['pixel_scores']])
        combined_edge_scores = np.concatenate([normal_results['edge_scores'], cutout_results['edge_scores']])
        combined_labels = np.concatenate([normal_results['labels'], cutout_results['labels']])
        
        cutout_pixel_metrics = compute_evaluation_metrics(combined_pixel_scores, combined_labels)
        cutout_edge_metrics = compute_evaluation_metrics(combined_edge_scores, combined_labels)
        
        metrics_summary["synthetic_sets"].append({
            "dataset_type": "cutout_synth",
            "num_normal": len(normal_results['pixel_scores']),
            "num_anomaly": len(cutout_results['pixel_scores']),
            "pixel_metrics": cutout_pixel_metrics,
            "edge_metrics": cutout_edge_metrics
        })
        
        create_reconstruction_gallery(
            cutout_results['original_images'][:5], 
            cutout_results['reconstructed_images'][:5],
            output_dir, 
            f"{class_name}_cutout"
        )
    
    if scratch_results:
        # we combine normal and scratch data
        combined_pixel_scores = np.concatenate([normal_results['pixel_scores'], scratch_results['pixel_scores']])
        combined_edge_scores = np.concatenate([normal_results['edge_scores'], scratch_results['edge_scores']])
        combined_labels = np.concatenate([normal_results['labels'], scratch_results['labels']])
        
        scratch_pixel_metrics = compute_evaluation_metrics(combined_pixel_scores, combined_labels)
        scratch_edge_metrics = compute_evaluation_metrics(combined_edge_scores, combined_labels)
        
        metrics_summary["synthetic_sets"].append({
            "dataset_type": "scratches_synth",
            "num_normal": len(normal_results['pixel_scores']),
            "num_anomaly": len(scratch_results['pixel_scores']),
            "pixel_metrics": scratch_pixel_metrics,
            "edge_metrics": scratch_edge_metrics
        })
        
        create_reconstruction_gallery(
            scratch_results['original_images'][:5], 
            scratch_results['reconstructed_images'][:5],
            output_dir, 
            f"{class_name}_scratch"
        )
    
    # final metrics summary
    with open(os.path.join(output_dir, "metrics", "metrics.json"), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE-based anomaly detection models")
    parser.add_argument("--exp_dir", type=str, required=True, 
                        help="Path to experiment directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (default: <exp_dir>/evaluation)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--testset", type=str, default="all",
                        choices=["all", "real", "synthetic"],
                        help="Type of test set to evaluate on")
    
    args = parser.parse_args()
    
    # Normalize all paths to use consistent separators
    if args.exp_dir:
        args.exp_dir = os.path.normpath(args.exp_dir)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_dir, "evaluation")
    
    args.output_dir = os.path.normpath(args.output_dir)  # Normalize output path
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, class_name = load_model(args.exp_dir)
    
    if args.testset == "all" or args.testset == "real":
        print(f"Evaluating on real test data for class: {class_name}")
        metrics_summary = evaluate_and_save_results(model, class_name, args.output_dir, args.batch_size)
        
        print("\n--- Evaluation Results ---")
        print(f"Class: {class_name}")
        
        for test_set in metrics_summary["test_sets"]:
            print(f"\nDataset: {test_set['dataset_type']}")
            print(f"  Pixel MAE - ROC-AUC: {test_set['pixel_metrics']['roc_auc']:.4f}")
            print(f"  Pixel MAE - AP: {test_set['pixel_metrics']['ap']:.4f}")
            # print(f"  Pixel MAE - Prec@5%FPR: {test_set['pixel_metrics']['precision_at_5pct']:.4f}")
            
            print(f"  Edge Score - ROC-AUC: {test_set['edge_metrics']['roc_auc']:.4f}")
            print(f"  Edge Score - AP: {test_set['edge_metrics']['ap']:.4f}")
            # print(f"  Edge Score - Prec@5%FPR: {test_set['edge_metrics']['precision_at_5pct']:.4f}")
    
    if args.testset == "all" or args.testset == "synthetic":
        if args.testset == "synthetic":
            print(f"\nEvaluating on synthetic test data for class: {class_name}")
            try:
                evaluate_dataset(
                    model, class_name, 
                    dataset_type='cutout_synth', 
                    is_anomaly=True,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size
                )
                print("Evaluated cutout synthetic dataset")
            except:
                print("Cutout synthetic dataset not found. Skipping...")
            
            try:
                evaluate_dataset(
                    model, class_name, 
                    dataset_type='scratches_synth', 
                    is_anomaly=True,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size
                )
                print("Evaluated scratches synthetic dataset")
            except:
                print("Scratches synthetic dataset not found. Skipping...")
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
