# for visualization tools to compare original/reconstructed images and edge maps

import os
import json
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def verify_visualization_inputs(run_dirs):
    """Verify if all required inputs for visualizations exist"""
    status = {"ready": True, "missing": []}
    
    for run_dir in run_dirs:
        # per-image scores
        per_image_scores_path = os.path.join(run_dir, "metrics", "per_image_scores_test.json")
        if not os.path.exists(per_image_scores_path):
            status["ready"] = False
            status["missing"].append(f"{per_image_scores_path} - per-image scores missing")
        
        # reconstructed images directory
        recon_dir = os.path.join(run_dir, "recon_examples")
        if not os.path.exists(recon_dir):
            status["ready"] = False
            status["missing"].append(f"{recon_dir} - reconstruction examples missing")
            
        # metrics.json
        metrics_path = os.path.join(run_dir, "metrics", "metrics.json")
        if not os.path.exists(metrics_path):
            status["ready"] = False
            status["missing"].append(f"{metrics_path} - metrics.json missing")
        else:
            # if contains required fields
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                required_fields = ["ap", "roc_auc", "precision_at_5pct"]
                for field in required_fields:
                    if field not in metrics.get("test_sets", [{}])[0].get("pixel_metrics", {}):
                        status["ready"] = False
                        status["missing"].append(f"{metrics_path} - missing required field: {field}")
                        
        # epoch_metrics.json
        epoch_metrics_path = os.path.join(run_dir, "metrics", "epoch_metrics.json")
        if not os.path.exists(epoch_metrics_path):
            status["missing"].append(f"{epoch_metrics_path} - epoch metrics missing (optional)")
    
    return status


def plot_pr_curves(runs, outpath):
    """
    plot Precision-Recall curves
    baseline vs improved
    
    runs: list of run directories
    outpath, for the PR curve figure
    """
    import matplotlib.pyplot as plt
    from sklearn import metrics as skmetrics
    
    plt.figure(figsize=(10, 6))
    
    for run_dir in runs:
        class_name = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
        model_type = os.path.basename(os.path.dirname(run_dir))
        run_id = os.path.basename(run_dir)
        
        # per-image scores
        scores_path = os.path.join(run_dir, "metrics", "per_image_scores_test.json")
        with open(scores_path, 'r') as f:
            scores_data = json.load(f)
        
        # scores and labels
        pixel_scores = np.array(scores_data["pixel_scores"])
        labels = np.array(scores_data["labels"])
        
        # precision-recall curve
        precision, recall, _ = skmetrics.precision_recall_curve(labels, pixel_scores)
        ap = skmetrics.average_precision_score(labels, pixel_scores)
        
        # plot it
        label = f"{model_type} (AP: {ap:.3f})"
        plt.plot(recall, precision, label=label, linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.title(f"Precision-Recall Curves for {class_name}")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()
    
    # save a copy to reports/figures directory
    report_path = os.path.join("reports", "figures", f"figure_pr_{class_name}.png")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    plt.savefig(report_path, dpi=300)


def build_metrics_table(run_dirs, out_csv):
    """
    build a compact metrics table for all runs
    
    run_dirs: list of run directories
    out_csv: output path
    """    
    data = []
    
    for run_dir in run_dirs:
        class_name = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
        model_type = os.path.basename(os.path.dirname(run_dir))
        run_id = os.path.basename(run_dir)
        
        # load metrics
        metrics_path = os.path.join(run_dir, "metrics", "metrics.json")
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        
        # get metrics from the real_test entry
        for test_set in metrics_data.get("test_sets", []):
            if test_set.get("dataset_type") == "real_test":
                pixel_metrics = test_set.get("pixel_metrics", {})
                
                # get lambda_geo value if it is improved model
                lambda_geo = "N/A"
                if "improved" in run_id:
                    lambda_parts = [p for p in run_id.split("_") if p.startswith("lambda")]
                    if lambda_parts:
                        lambda_geo = lambda_parts[0].replace("lambda", "")
                
                row = {
                    "Run ID": run_id,
                    "Class": class_name,
                    "Model": "Improved" if "improved" in model_type else "Baseline",
                    "Lambda_geo": lambda_geo,
                    "AP": pixel_metrics.get("ap", float("nan")),
                    "ROC-AUC": pixel_metrics.get("roc_auc", float("nan")),
                    "Precision@5%FPR": pixel_metrics.get("precision_at_5pct", float("nan"))
                }
                data.append(row)
    
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    # png fromat
    out_png = out_csv.replace(".csv", ".png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def make_compact_gallery(run_dir, class_name, output_path, threshold=None):
    """
    Create a compact gallery of reconstruction examples, for TP, FP, and FN examples.
    
    run_dir, experiment run directory
    class_name
    output_path, for the gallery image
    threshold is the optional threshold for classification (if == None  then use optimal threshold)
    """
    scores_path = os.path.join(run_dir, "metrics", "per_image_scores_test.json")
    with open(scores_path, 'r') as f:
        scores_data = json.load(f)
    
    scores = np.array(scores_data["pixel_scores"])
    labels = np.array(scores_data["labels"])
    
    # if threshold == None
    if threshold is None:
        metrics_path = os.path.join(run_dir, "metrics", "metrics.json")
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        for test_set in metrics_data.get("test_sets", []):
            if test_set.get("dataset_type") == "real_test":
                threshold = test_set.get("pixel_metrics", {}).get("optimal_threshold", 0.5)
    
    predictions = (scores >= threshold).astype(int)
    
    tp_indices = np.where((predictions == 1) & (labels == 1))[0]
    fp_indices = np.where((predictions == 1) & (labels == 0))[0]
    fn_indices = np.where((predictions == 0) & (labels == 1))[0]
    
    # 4 TP, 3 FP, 3 FN
    selected_indices = []
    selected_indices.extend(tp_indices[:4] if len(tp_indices) >= 4 else tp_indices)
    selected_indices.extend(fp_indices[:3] if len(fp_indices) >= 3 else fp_indices)
    selected_indices.extend(fn_indices[:3] if len(fn_indices) >= 3 else fn_indices)
    
    # if we don't have enough, we fill with random samples
    if len(selected_indices) < 10:
        remaining = 10 - len(selected_indices)
        available = np.setdiff1d(np.arange(len(scores)), selected_indices)
        if len(available) >= remaining:
            selected_indices.extend(np.random.choice(available, remaining, replace=False))
    
    # limit 10
    selected_indices = selected_indices[:10]
    
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 5, figure=fig)
    
    for i, idx in enumerate(selected_indices):
        row = i // 5
        col = i % 5
        
        image_id = scores_data.get("image_ids", [f"image_{i}" for i in range(len(scores))])[idx]
        orig_path = os.path.join(run_dir, "recon_examples", f"{image_id}_orig.png")
        recon_path = os.path.join(run_dir, "recon_examples", f"{image_id}_recon.png")
        diff_path = os.path.join(run_dir, "recon_examples", f"{image_id}_diff.png")
        edge_orig_path = os.path.join(run_dir, "recon_examples", f"{image_id}_edge_orig.png")
        edge_recon_path = os.path.join(run_dir, "recon_examples", f"{image_id}_edge_recon.png")
        
        ax = fig.add_subplot(gs[row, col])
        
        if os.path.exists(orig_path) and os.path.exists(recon_path):
            from PIL import Image
            
            orig_img = np.array(Image.open(orig_path))
            recon_img = np.array(Image.open(recon_path))
            
            if os.path.exists(diff_path):
                diff_img = np.array(Image.open(diff_path))
            else:
                diff_img = np.abs(orig_img - recon_img)
                diff_img = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min() + 1e-8)
                diff_img = (diff_img * 255).astype(np.uint8)
            
            # orig | recon | diff | edge_orig | edge_recon
            height = orig_img.shape[0]
            composite = np.zeros((height * 5, orig_img.shape[1], 3), dtype=np.uint8)
            
            # add original
            composite[:height] = orig_img if len(orig_img.shape) == 3 else np.stack([orig_img]*3, axis=2)
            
            # reconstruction
            composite[height:2*height] = recon_img if len(recon_img.shape) == 3 else np.stack([recon_img]*3, axis=2)
            
            #difference
            composite[2*height:3*height] = diff_img if len(diff_img.shape) == 3 else np.stack([diff_img]*3, axis=2)
            
            # edge maps if exist
            if os.path.exists(edge_orig_path) and os.path.exists(edge_recon_path):
                edge_orig = np.array(Image.open(edge_orig_path))
                edge_recon = np.array(Image.open(edge_recon_path))
                
                composite[3*height:4*height] = edge_orig if len(edge_orig.shape) == 3 else np.stack([edge_orig]*3, axis=2)
                composite[4*height:] = edge_recon if len(edge_recon.shape) == 3 else np.stack([edge_recon]*3, axis=2)
            
            ax.imshow(composite)
            
            prediction = "Anomaly" if predictions[idx] == 1 else "Normal"
            true_label = "Anomaly" if labels[idx] == 1 else "Normal"
            category = "TP" if predictions[idx] == 1 and labels[idx] == 1 else \
                       "FP" if predictions[idx] == 1 and labels[idx] == 0 else \
                       "FN" if predictions[idx] == 0 and labels[idx] == 1 else "TN"
            
            ax.set_title(f"{category}: pred={prediction}, true={true_label}\nscore={scores[idx]:.4f}")
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    report_path = os.path.join("reports", "figures", f"gallery_{class_name}.png")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(epoch_metrics_path, outpath):
    """
    plot loss curves from training epochs.
    
    epoch_metrics_path
    outpath
    """
    import matplotlib.pyplot as plt
    
    with open(epoch_metrics_path, 'r') as f:
        epoch_data = json.load(f)
    
    # get loss arrays
    epochs = range(1, len(epoch_data.get("recon_loss", [])) + 1)
    recon_loss = epoch_data.get("recon_loss", [])
    kl_loss = epoch_data.get("kl_loss", [])
    geo_loss = epoch_data.get("geo_loss", [])
    total_loss = epoch_data.get("total_loss", [])
    
    plt.figure(figsize=(10, 6))
    
    if recon_loss:
        plt.plot(epochs, recon_loss, label='Reconstruction Loss')
    if kl_loss:
        plt.plot(epochs, kl_loss, label='KL Divergence Loss')
    if geo_loss:
        plt.plot(epochs, geo_loss, label='Geometric Prior Loss')
    if total_loss:
        plt.plot(epochs, total_loss, label='Total Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.title('Training Loss Curves')
    plt.tight_layout()
    
    # save
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_score_hist_and_scatter(scores_path, out_hist, out_scatter):
    """
    histogram of scores and scatter plot of MAE vs edge scores.
    
    scores_path
    out_hist, for histogram figure
    out_scatter
    """
    with open(scores_path, 'r') as f:
        scores_data = json.load(f)
    
    pixel_scores = np.array(scores_data["pixel_scores"])
    edge_scores = np.array(scores_data["edge_scores"])
    labels = np.array(scores_data["labels"])
    
    plt.figure(figsize=(10, 6))
    
    # normal samples
    plt.hist(pixel_scores[labels == 0], bins=30, alpha=0.5, label='Normal', color='green')
    # anomaly
    plt.hist(pixel_scores[labels == 1], bins=30, alpha=0.5, label='Anomaly', color='red')
    
    plt.xlabel('MAE Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Score Distribution: Normal vs Anomaly')
    plt.tight_layout()
    
    # save
    os.makedirs(os.path.dirname(out_hist), exist_ok=True)
    plt.savefig(out_hist, dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    normal = plt.scatter(pixel_scores[labels == 0], edge_scores[labels == 0], 
                        c='green', alpha=0.6, label='Normal')
    anomaly = plt.scatter(pixel_scores[labels == 1], edge_scores[labels == 1], 
                         c='red', alpha=0.6, label='Anomaly')
    
    plt.xlabel('MAE Score')
    plt.ylabel('Edge Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('MAE vs Edge Scores')
    plt.tight_layout()
    
    # save
    os.makedirs(os.path.dirname(out_scatter), exist_ok=True)
    plt.savefig(out_scatter, dpi=300)
    plt.close()
    

def save_confusion_matrix(scores_path, threshold, out_png, out_csv):
    """
    confusion matrix at a given threshold.
    
    scores_path: for per-image scores JSON
    threshold for Classification
    out_png
    out_csv
    """    

    with open(scores_path, 'r') as f:
        scores_data = json.load(f)
    
    scores = np.array(scores_data["pixel_scores"])# pixel-based MAE scores
    labels = np.array(scores_data["labels"])
    
    predictions = (scores >= threshold).astype(int)
    
    cm = confusion_matrix(labels, predictions)
    
    # save
    df_cm = pd.DataFrame(
        cm, 
        index=['Actual Normal', 'Actual Anomaly'], 
        columns=['Predicted Normal', 'Predicted Anomaly']
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_cm.to_csv(out_csv)
    
    # confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix (threshold={threshold:.4f})')
    plt.tight_layout()
    
    # save
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
