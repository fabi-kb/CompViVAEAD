# for visualization tools to compare original/reconstructed images and edge maps

import os
import json
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


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

