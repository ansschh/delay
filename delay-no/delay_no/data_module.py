import torch
import pytorch_lightning as pl
from typing import Optional, Tuple
from delay_no.datasets import create_data_module


class DelayDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Delay Neural Operator datasets"""
    
    def __init__(self, model_config, data_config):
        """
        Initialize DataModule with model and data configs
        
        Parameters:
        -----------
        model_config : dict
            Model configuration
        data_config : dict
            Data configuration
        """
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.train_loader = None
        self.val_loader = None
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup train and validation dataloaders
        
        Parameters:
        -----------
        stage : str, optional
            Stage - 'fit', 'validate', 'test', or 'predict'
        """
        self.train_loader, self.val_loader = create_data_module(
            self.model_config.get("variant", "stacked"),
            self.data_config
        )
        
    def train_dataloader(self):
        """Return training dataloader"""
        return self.train_loader
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return self.val_loader
