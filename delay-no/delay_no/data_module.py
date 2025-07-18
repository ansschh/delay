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
        
        # Ensure consistency between model and dataset parameters
        data_config = dict(self.data_config)
        
        # If model config has parameters that should be consistent with dataset, use those values
        if "S" in self.model_config and "S" in data_config:
            if self.model_config["S"] != data_config["S"]:
                logger.warning(f"Model S={self.model_config['S']} doesn't match data S={data_config['S']}. Using model value.")
                data_config["S"] = self.model_config["S"]
                
        # If model defines in_ch/out_ch, ensure dataset nx is consistent
        if "in_ch" in self.model_config and "nx" in data_config:
            if self.model_config["in_ch"] != data_config["nx"]:
                logger.warning(f"Model in_ch={self.model_config['in_ch']} doesn't match data nx={data_config['nx']}. Using model value.")
                data_config["nx"] = self.model_config["in_ch"]
        
        # Create dataloaders
        self.train_loader, self.val_loader = create_data_module(
            variant,
            data_config
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
