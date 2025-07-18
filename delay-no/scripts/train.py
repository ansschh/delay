#!/usr/bin/env python
# Disable TensorFlow/TensorBoard warnings and errors right at the start
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logging
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::DeprecationWarning'  # Ignore warnings
import sys
import torch

# Set PyTorch to use Tensor Cores efficiently
torch.set_float32_matmul_precision('high')

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger  # Disabled to avoid import errors
from omegaconf import DictConfig, OmegaConf
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delay_no.data_module import DelayDataModule
from delay_no.models.stacked import StackedFNO
from delay_no.models.steps import StepsLit
from delay_no.models.kernel import KernelLit

logger = logging.getLogger(__name__)


def get_model(config):
    """
    Initialize model based on configuration
    
    Parameters:
    -----------
    config : DictConfig
        Model configuration
        
    Returns:
    --------
    pl.LightningModule: Model instance
    """
    model_type = config.model.name
    
    if model_type == "stacked":
        # Convert n_modes from Hydra ListConfig to tuple
        n_modes = tuple(config.model.n_modes) if hasattr(config.model.n_modes, '__iter__') else config.model.n_modes
        
        model = StackedFNO(
            in_ch=config.model.in_ch,
            out_ch=config.model.out_ch,
            S=config.model.S,
            n_modes=n_modes,
            hidden=config.model.hidden_dim,
            L=config.model.n_layers,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
    elif model_type == "steps":
        model = StepsLit(
            hist_dim=config.model.hist_dim,
            tau_dim=config.model.tau_dim,
            out_len=config.model.out_len,
            S=config.model.S,
            n_modes=config.model.n_modes[0],  # Only using 1D modes
            K=config.model.K,
            spectral_penalty=config.model.spectral_penalty,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
    elif model_type == "kernel":
        model = KernelLit(
            S=config.model.S,
            in_ch=config.model.in_ch,
            out_ch=config.model.out_ch,
            hidden=config.model.hidden_dim,
            n_spatial_modes=config.model.n_spatial_modes,  # None for 1D problems
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def train(config: DictConfig):
    """
    Main training function
    
    Parameters:
    -----------
    config : DictConfig
        Hydra configuration
    """
    pl.seed_everything(config.train.seed)
    
    # Configure paths
    run_dir = os.getcwd()
    logger.info(f"Run directory: {run_dir}")
    
    # Setup data module
    data_module = DelayDataModule(
        model_config=config.model,
        data_config=config.data
    )
    
    # Initialize model
    model = get_model(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename=f"{config.model.name}-{config.data.family}" + "-{epoch}-{val_l2:.4f}",
        monitor="val_l2",
        save_top_k=3,
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    callbacks = [checkpoint_callback, lr_monitor]
    
    # Use only default logger to avoid import/compatibility issues
    # Disable all external loggers (WandB, TensorBoard, etc.) for now
    logger_to_use = True  # Use default PyTorch Lightning logger
    print("Using default PyTorch Lightning logger (external loggers disabled)")
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger_to_use,
        callbacks=callbacks,
        precision=config.train.precision,
        gradient_clip_val=config.train.gradient_clip,
        log_every_n_steps=config.logging.log_steps,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save final checkpoint
    trainer.save_checkpoint(os.path.join(run_dir, "checkpoints", "final.ckpt"))
    
    return {
        "best_val_l2": checkpoint_callback.best_model_score.item(),
        "checkpoint_path": checkpoint_callback.best_model_path
    }


if __name__ == "__main__":
    train()
