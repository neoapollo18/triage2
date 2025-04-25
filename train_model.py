import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import glob
import os
import wandb
from _3pl_matching_model import run_once

def get_latest_matches():
    match_files = glob.glob('labeled_data.csv')
    if not match_files:
        raise FileNotFoundError("No match files found!")
    return max(match_files, key=lambda x: Path(x).stat().st_mtime)

def repeat_cv(bus, tpl, labeled):
    """Run 5 repetitions of 5-fold CV and log results"""
    wandb.init(project="3pl-matching", name=f"small_{datetime.now():%Y%m%d_%H%M%S}")
    all_means = []
    
    print("\nStarting 5 repetitions of 5-fold CV...")
    for rep in range(5):
        print(f"\nRepetition {rep+1}/5")
        m, s = run_once(bus, tpl, labeled, rep)
        print(f"Rep {rep+1} - Mean AUC: {m:.4f} ± {s:.4f}")
        
        wandb.log({
            "rep": rep + 1,
            "mean_auc": m,
            "std_auc": s
        })
        all_means.append(m)
    
    final_mean = float(np.mean(all_means))
    final_std = float(np.std(all_means))
    
    print("\nFinal Results:")
    print(f"Mean AUC across all repetitions: {final_mean:.4f} ± {final_std:.4f}")
    
    wandb.summary["final_mean_auc"] = final_mean
    wandb.summary["final_std_auc"] = final_std
    wandb.finish()

if __name__ == "__main__":
    print("Loading datasets...")
    
    # Load the base datasets
    businesses = pd.read_csv('businesses.csv')
    threepls = pd.read_csv('3pls.csv')
    
    # Load the most recent matches
    latest_matches = get_latest_matches()
    print(f"Using matches from: {latest_matches}")
    labeled_data = pd.read_csv(latest_matches)
    
    print(f"Loaded {len(businesses)} businesses, {len(threepls)} 3PLs, and {len(labeled_data)} labeled matches")
    
    # Run the repeated cross-validation
    print("\nStarting training with small model architecture...")
    repeat_cv(
        bus=businesses,
        tpl=threepls,
        labeled=labeled_data
    )
    
    print("\nTraining complete! Check wandb for detailed metrics and visualizations.") 