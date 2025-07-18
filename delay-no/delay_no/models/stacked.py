import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..layers.spectral_conv_nd import SpectralConvND
from ..layers.channel_mlp import ChannelMLP

class StackedFNO(pl.LightningModule):
    """
    Stacked-history FNO (Variant A)
    Treats history as an additional dimension and solves the PDE in (x, s) space
    """
    def __init__(self, in_ch=1, out_ch=1, S=16, n_modes=(4, 16), hidden=64, L=4, 
                 lr=3e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden = hidden
        self.n_modes = n_modes
        self.S = S
        self.L = L
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Model components
        # For now we use nx from the dataset as input channels
        # Dynamic channel handling - will be adjusted based on actual input
        # We add 2 for the coordinate channels (s, x)
        self.lift = ChannelMLP(in_ch + 2, hidden, hidden)  # +2 for (s,x) coords
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            nn.Sequential(
                SpectralConvND(hidden, hidden, n_modes),
                nn.Conv2d(hidden, hidden, 1),  # point-wise W
                nn.GELU()
            ) for _ in range(L)
        ])
        
        self.proj = ChannelMLP(hidden, hidden, out_ch)

    def add_coords(self, h):
        """Add normalized coordinate channels to input"""
        B, C, S, X = h.shape
        s = torch.linspace(-1, 0, S, device=h.device)[None, None, :, None].expand(B, 1, S, X)
        x = torch.linspace(0, 1, X, device=h.device)[None, None, None, :].expand(B, 1, S, X)
        return torch.cat([h, s, x], dim=1)

    def forward(self, hist):
        """
        Forward pass through the model
        
        Parameters:
        -----------
        hist : torch.Tensor
            History tensor of shape (B, C, S, X)
            
        Returns:
        --------
        torch.Tensor: Predicted next window of shape (B, C, S, X)
        """
        # Get actual input shape and adapt the model if this is the first forward pass
        B, C, S, X = hist.shape
        
        # Handle channel dimension mismatch by recreating the lift layer if needed
        # This allows the model to adapt to the actual channel count at runtime
        if hasattr(self, '_first_forward') and self._first_forward:
            if C != self.in_ch:
                # Update in_ch to match actual input
                self.in_ch = C
                # Recreate lift layer with correct input channel count
                self.lift = ChannelMLP(C + 2, self.hidden, self.hidden).to(hist.device)
                print(f"Adjusted lift layer input channels from {self.in_ch} to {C}")
            self._first_forward = False
        else:
            self._first_forward = True
        
        # Add coordinate channels
        z = self.add_coords(hist)  # (B, C+2, S, X)
        
        # Permute to move channels to last dimension for MLP
        z_perm = z.permute(0, 2, 3, 1)  # (B, S, X, C+2)
        
        # Lift to hidden dimension
        z = self.lift(z_perm)  # (B, S, X, hidden)
        
        # Move channels back to second dim for convolution
        z = z.permute(0, 3, 1, 2)  # (B, hidden, S, X)
        
        # Apply FNO blocks with checkpointing if available
        for blk in self.fno_blocks:
            if self.training and hasattr(torch.utils, 'checkpoint'):
                z = torch.utils.checkpoint.checkpoint(blk, z)
            else:
                z = blk(z)
        
        # Project back to output dimension
        out = self.proj(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return out
    
    def training_step(self, batch, batch_idx):
        """Lightning training step"""
        # Handle different input shapes - reshape to (B, C, S, X)
        hist = batch["hist"]
        if len(hist.shape) == 3:  # (B, nx, S)
            # Add spatial dimension X=1
            hist = hist.unsqueeze(-1)  # (B, nx, S, 1)
        
        target = batch["y"]
        if len(target.shape) == 3:  # (B, nx, T)
            # Add spatial dimension X=1
            target = target.unsqueeze(-1)  # (B, nx, T, 1)
        
        # Forward pass
        pred = self.forward(hist)
        
        # Calculate loss
        mse_loss = torch.nn.functional.mse_loss(pred, target)
        
        # Optional: spectral loss component
        if hasattr(self, "spectral_weight") and self.spectral_weight > 0:
            pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target, dim=(-2, -1))
            spectral_loss = torch.nn.functional.mse_loss(pred_fft.abs(), target_fft.abs())
            loss = mse_loss + 0.1 * spectral_loss
            self.log("train_spectral_loss", spectral_loss)
        else:
            loss = mse_loss
            
        self.log("train_loss", loss)
        self.log("train_mse", mse_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step"""
        # Handle different input shapes - reshape to (B, C, S, X)
        hist = batch["hist"]
        if len(hist.shape) == 3:  # (B, nx, S)
            # Add spatial dimension X=1
            hist = hist.unsqueeze(-1)  # (B, nx, S, 1)
        
        target = batch["y"]
        if len(target.shape) == 3:  # (B, nx, T)
            # Add spatial dimension X=1
            target = target.unsqueeze(-1)  # (B, nx, T, 1)
        
        # Forward pass
        pred = self.forward(hist)
        
        # Calculate loss
        mse_loss = torch.nn.functional.mse_loss(pred, target)
        l2_loss = torch.sqrt(mse_loss)
        
        self.log("val_mse", mse_loss)
        self.log("val_l2", l2_loss)
        
        return {"val_loss": mse_loss, "val_l2": l2_loss}
    
    def configure_optimizers(self):
        """Configure optimizers for training"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_l2"
            }
        }
