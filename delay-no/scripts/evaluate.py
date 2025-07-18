#!/usr/bin/env python3
import os
import sys
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import seaborn as sns
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delay_no.datasets import create_dataloaders
from delay_no.models.stacked import StackedFNO
from delay_no.models.steps import StepsLit
from delay_no.models.kernel import KernelLit
from delay_no.metrics import l2_relative_error, spectral_l2_error, energy_ratio, wall_clock

logger = logging.getLogger(__name__)


def load_model(config, checkpoint_path=None):
    """
    Load model from checkpoint or create new model
    
    Parameters:
    -----------
    config : DictConfig
        Model configuration
    checkpoint_path : str, optional
        Path to checkpoint file
        
    Returns:
    --------
    pl.LightningModule: Model instance
    """
    model_type = config.model.name
    
    if model_type == "stacked":
        ModelClass = StackedFNO
    elif model_type == "steps":
        ModelClass = StepsLit
    elif model_type == "kernel":
        ModelClass = KernelLit
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = ModelClass.load_from_checkpoint(checkpoint_path)
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, creating new model from config")
        if model_type == "stacked":
            model = StackedFNO(
                in_ch=config.model.in_ch,
                out_ch=config.model.out_ch,
                S=config.model.S,
                n_modes=config.model.n_modes,
                hidden=config.model.hidden_dim,
                L=config.model.n_layers
            )
        elif model_type == "steps":
            model = StepsLit(
                hist_dim=config.model.hist_dim,
                tau_dim=config.model.tau_dim,
                out_len=config.model.out_len,
                S=config.model.S,
                n_modes=config.model.n_modes[0],
                K=config.model.K,
                spectral_penalty=config.model.spectral_penalty
            )
        elif model_type == "kernel":
            model = KernelLit(
                S=config.model.S,
                in_ch=config.model.in_ch,
                out_ch=config.model.out_ch,
                hidden=config.model.hidden_dim,
                n_spatial_modes=config.model.n_spatial_modes
            )
    
    return model


def plot_predictions(samples, predictions, targets, save_dir, family, model_name):
    """
    Plot predictions vs targets for selected samples
    
    Parameters:
    -----------
    samples : list
        List of sample indices
    predictions : torch.Tensor
        Predictions tensor
    targets : torch.Tensor
        Targets tensor
    save_dir : str
        Directory to save plots
    family : str
        Dataset family name
    model_name : str
        Model name
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i in samples:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data based on dimensions
        if predictions.dim() > 3:
            # 2D spatial data (e.g., Reaction-Diffusion)
            pred = predictions[i, 0].detach().cpu().numpy()
            targ = targets[i, 0].detach().cpu().numpy()
            
            im0 = axes[0].imshow(pred, cmap='viridis')
            im1 = axes[1].imshow(targ, cmap='viridis')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            # 1D data
            t = np.arange(predictions.shape[-1])
            pred = predictions[i].squeeze().detach().cpu().numpy()
            targ = targets[i].squeeze().detach().cpu().numpy()
            
            axes[0].plot(t, pred)
            axes[1].plot(t, targ)
        
        axes[0].set_title(f"Prediction (Sample {i})")
        axes[1].set_title(f"Target (Sample {i})")
        
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{family}_{model_name}_sample_{i}.png"), dpi=150)
        plt.close()


def plot_error_histogram(errors, save_path, title):
    """
    Plot histogram of errors
    
    Parameters:
    -----------
    errors : list
        List of error values
    save_path : str
        Path to save plot
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(title)
    plt.xlabel("Relative L2 Error")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_comparison_table(results, save_path):
    """
    Create and save comparison table
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    save_path : str
        Path to save table
    """
    # Create dataframe
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    return df


def plot_rollout_stability(model, sample_batch, steps, save_path, device='cuda'):
    """
    Plot rollout stability for a given model
    
    Parameters:
    -----------
    model : pl.LightningModule
        Model to evaluate
    sample_batch : dict or tuple
        Sample batch for rollout
    steps : int
        Number of rollout steps
    save_path : str
        Path to save plot
    device : str
        Device to use for computation
    """
    model = model.to(device)
    model.eval()
    
    # Prepare input data
    if isinstance(sample_batch, dict):
        hist = sample_batch["hist"].to(device)
        if "u0" in sample_batch:
            u0 = sample_batch["u0"].to(device)
        else:
            u0 = hist[:, :, -1]
    else:
        hist, tau, _ = [t.to(device) for t in sample_batch]
    
    # Perform rollout based on model type
    with torch.no_grad():
        if isinstance(model, StackedFNO):
            preds = model(hist)
            energies = [torch.mean(hist**2).item(), torch.mean(preds**2).item()]
        
        elif isinstance(model, StepsLit):
            # Multiple step rollout
            preds_list = []
            h = hist
            
            for i in range(steps):
                pred = model(h, tau)
                preds_list.append(pred[:, -1, :].unsqueeze(1))
                # Update history by shifting and appending prediction
                h = torch.cat([h[:, :, model.Δt_idx:], pred[:, -1, :].unsqueeze(1).unsqueeze(2)], dim=-1)
            
            preds = torch.cat(preds_list, dim=1)
            energies = [torch.mean(hist**2).item()]
            for i in range(len(preds_list)):
                energies.append(torch.mean(preds_list[i]**2).item())
        
        elif isinstance(model, KernelLit):
            # Integrate for multiple steps
            dt = 0.01  # Default if not provided
            if isinstance(sample_batch, dict) and "dt" in sample_batch:
                dt = sample_batch["dt"].item()
                
            traj = model(hist, u0, dt, steps=steps)
            energies = [torch.mean(u0**2).item()]
            for i in range(1, traj.shape[1]):
                energies.append(torch.mean(traj[:, i]**2).item())
    
    # Plot energy evolution
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(energies)), energies, 'o-')
    plt.axhline(y=energies[0], color='r', linestyle='--', label='Initial Energy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Energy Evolution During Rollout")
    plt.xlabel("Rollout Step")
    plt.ylabel("Energy (Mean Squared)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return energies


@hydra.main(config_path="../configs", config_name="train")
def evaluate(config: DictConfig):
    """
    Main evaluation function
    
    Parameters:
    -----------
    config : DictConfig
        Hydra configuration
    """
    # Configure paths
    run_dir = os.getcwd()
    logger.info(f"Run directory: {run_dir}")
    
    # Define checkpoint path
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        # Find best checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt") and "final" not in f]
        if checkpoints:
            checkpoints.sort(key=lambda x: float(x.split("val_l2=")[-1].split("-")[0]))
            best_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
        else:
            best_checkpoint = os.path.join(checkpoint_dir, "final.ckpt")
    else:
        best_checkpoint = None
    
    # Load model
    model = load_model(config, best_checkpoint)
    model.eval()
    
    # Setup dataloaders
    _, val_dl = create_dataloaders(
        dataset_path=config.data.path,
        family=config.data.family,
        variant=config.model.name,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=False,
        grid_size=config.model.S,
        pin_memory=True
    )
    
    # Create results directory
    results_dir = os.path.join(run_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluation metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    l2_errors = []
    spectral_errors = []
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dl):
            # Forward pass
            if isinstance(batch, dict):
                # Handle different batch formats based on model type
                if isinstance(model, StackedFNO):
                    hist = batch["hist"].to(device)
                    target = batch["y"].to(device)
                    
                    # Ensure correct dimensionality
                    if len(hist.shape) == 3:
                        hist = hist.unsqueeze(1)
                    if len(target.shape) == 3:
                        target = target.unsqueeze(1)
                    
                    pred = model(hist)
                
                elif isinstance(model, KernelLit):
                    hist = batch["hist"].to(device)
                    u0 = batch["u0"].to(device)
                    target = batch["target"].to(device)
                    dt = batch["dt"].item()
                    
                    # Determine integration steps based on target
                    steps = target.shape[1] - 1 if len(target.shape) > 2 else 1
                    traj = model(hist, u0, dt, steps)
                    
                    # Match output format to target
                    if len(target.shape) > 2:
                        pred = traj[:, 1:, :]  # Skip initial condition
                    else:
                        pred = traj[:, -1, :]  # Take only final state
            
            else:  # Tuple format for StepsLit
                h0, tau, gt = [t.to(device) for t in batch]
                pred = model(h0, tau)
                target = gt
            
            # Calculate metrics
            l2_err = l2_relative_error(pred, target)
            spec_err = spectral_l2_error(pred, target)
            
            l2_errors.append(l2_err)
            spectral_errors.append(spec_err)
            
            # Store some predictions for visualization
            if batch_idx < 5:
                predictions_list.append(pred)
                targets_list.append(target)
    
    # Compute average metrics
    avg_l2_error = np.mean(l2_errors)
    avg_spectral_error = np.mean(spectral_errors)
    
    # Measure throughput
    throughput = wall_clock(model, next(iter(val_dl)), runs=20, device=device)
    
    # Stability analysis (for a single batch)
    stability_path = os.path.join(results_dir, f"{config.model.name}_{config.data.family}_stability.png")
    sample_batch = next(iter(val_dl))
    energy_evolution = plot_rollout_stability(model, sample_batch, steps=10, save_path=stability_path, device=device)
    
    # Compute stability metric (ratio of final to initial energy)
    stability_metric = energy_evolution[-1] / energy_evolution[0]
    
    # Visualize predictions
    predictions = torch.cat(predictions_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    plot_predictions(
        samples=range(min(5, predictions.shape[0])), 
        predictions=predictions, 
        targets=targets, 
        save_dir=results_dir,
        family=config.data.family,
        model_name=config.model.name
    )
    
    # Plot error histogram
    error_hist_path = os.path.join(results_dir, f"{config.model.name}_{config.data.family}_error_hist.png")
    plot_error_histogram(l2_errors, error_hist_path, f"{config.model.name} on {config.data.family} - L2 Error Distribution")
    
    # Create results dictionary
    results = {
        "model": config.model.name,
        "dataset": config.data.family,
        "avg_l2_error": avg_l2_error,
        "avg_spectral_error": avg_spectral_error,
        "stability_metric": stability_metric,
        "throughput": throughput,
        "model_parameters": sum(p.numel() for p in model.parameters())
    }
    
    # Save results to CSV
    results_path = os.path.join(results_dir, f"{config.model.name}_{config.data.family}_results.csv")
    create_comparison_table([results], results_path)
    
    # Print summary
    logger.info(f"Evaluation Results for {config.model.name} on {config.data.family}:")
    logger.info(f"  Average L2 Error: {avg_l2_error:.6f}")
    logger.info(f"  Average Spectral Error: {avg_spectral_error:.6f}")
    logger.info(f"  Stability Metric: {stability_metric:.6f}")
    logger.info(f"  Throughput: {throughput:.2f} samples/sec")
    logger.info(f"  Results saved to {results_dir}")
    
    return results


if __name__ == "__main__":
    evaluate()
