import torch
import pytorch_lightning as pl
from typing import Optional, Tuple
import logging
from delay_no.datasets import create_data_module

logger = logging.getLogger(__name__)


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
        # Get variant from config
        variant = self.model_config.get("name", self.model_config.get("variant", "stacked"))
        
        # Create dataloaders
        self.train_loader, self.val_loader = create_data_module(
            variant,
            self.data_config
        )
        
        # Log dataloader status
        if self.train_loader is None:
            raise ValueError(f"No training data found for {self.data_config.get('family')} dataset")
            
        if self.val_loader is None:
            logger.warning(f"No validation data found for {self.data_config.get('family')} dataset. Using training data for validation.")
            self.val_loader = self.train_loader
        
    def train_dataloader(self):
        """Return training dataloader"""
        return self.train_loader
    
    def val_dataloader(self):
        """Return validation dataloader"""
        if self.val_loader is None:
            # If no validation loader is available, use training loader
            return self.train_loader
        return self.val_loader
