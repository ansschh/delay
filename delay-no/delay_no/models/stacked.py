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
        
        # FNO blocks with explicit dtype handling
        self.fno_blocks = nn.ModuleList()
        for _ in range(L):
            # Wrap spectral conv in a custom block to ensure proper dtype handling
            spectral_conv = SpectralConvND(hidden, hidden, n_modes)
            
            # Create sequential block with explicit type handling
            fno_block = nn.Sequential(
                spectral_conv,
                nn.Conv2d(hidden, hidden, 1),  # point-wise W
                nn.GELU()
            )
            
            self.fno_blocks.append(fno_block)
        
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
        # Get actual input shape and adapt the model if necessary
        B, C, S, X = hist.shape
        
        # Initialize the _first_forward flag if it doesn't exist
        if not hasattr(self, '_first_forward'):
            self._first_forward = True
            
        # Handle channel dimension mismatch by recreating the lift layer if needed
        # This allows the model to adapt to the actual channel count at runtime
        if self._first_forward:
            print(f"First forward pass with input shape: {hist.shape}, in_ch: {self.in_ch}")
            if C != self.in_ch:
                # Update in_ch to match actual input
                print(f"Channel mismatch detected: Expected {self.in_ch}, got {C}")
                old_in_ch = self.in_ch
                self.in_ch = C
                # Recreate lift layer with correct input channel count
                self.lift = ChannelMLP(C + 2, self.hidden, self.hidden).to(hist.device)
                print(f"Adjusted lift layer input channels from {old_in_ch} to {C}")
            self._first_forward = False
        
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
        """Lightning training step with improved shape handling"""
        try:
            # Get data and print shapes (on first batch)
            hist = batch["hist"]  # Expected: (B, S, nx) from pad_collate
            target = batch["y"]   # Expected: (B, T, nx) from pad_collate
            
            if batch_idx == 0:
                print(f"[training_step] Input shapes: hist={hist.shape}, target={target.shape}")
            
            # Reshape to (B, C, S, X) format expected by forward pass
            # Move channels (nx) to second dimension
            hist = hist.permute(0, 2, 1)  # (B, nx, S)
            
            # Add final dimension for 2D convolution operations
            if len(hist.shape) == 3:  # (B, nx, S)
                hist = hist.unsqueeze(-1)  # (B, nx, S, 1)
            
            # Similarly process target
            target = target.permute(0, 2, 1)  # (B, nx, T)
            if len(target.shape) == 3:
                target = target.unsqueeze(-1)  # (B, nx, T, 1)
                
            if batch_idx == 0:
                print(f"[training_step] Reshaped: hist={hist.shape}, target={target.shape}")
            
            # Forward pass
            pred = self.forward(hist)
            
            if batch_idx == 0:
                print(f"[training_step] Output shape: pred={pred.shape}")
            
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
            
        except Exception as e:
            print(f"Error in training_step: {str(e)}")
            print(f"Input shapes: hist={hist.shape if 'hist' in locals() else 'N/A'}, "
                  f"target={target.shape if 'target' in locals() else 'N/A'}")
            raise
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step with improved shape handling"""
        try:
            # Get data from batch
            hist = batch["hist"]  # Expected: (B, S, nx) from pad_collate
            target = batch["y"]   # Expected: (B, T, nx) from pad_collate
            
            if batch_idx == 0:
                print(f"[validation_step] Input shapes: hist={hist.shape}, target={target.shape}")
            
            # Reshape to (B, C, S, X) format expected by forward pass
            # Move channels (nx) to second dimension
            hist = hist.permute(0, 2, 1)  # (B, nx, S)
            
            # Add final dimension for 2D convolution operations
            if len(hist.shape) == 3:  # (B, nx, S)
                hist = hist.unsqueeze(-1)  # (B, nx, S, 1)
            
            # Similarly process target
            target = target.permute(0, 2, 1)  # (B, nx, T)
            if len(target.shape) == 3:
                target = target.unsqueeze(-1)  # (B, nx, T, 1)
                
            if batch_idx == 0:
                print(f"[validation_step] Reshaped: hist={hist.shape}, target={target.shape}")
            
            # Forward pass
            pred = self.forward(hist)
            
            if batch_idx == 0:
                print(f"[validation_step] Output shape: pred={pred.shape}")
        except Exception as e:
            print(f"Error in validation_step: {str(e)}")
            print(f"Input shapes: hist={hist.shape if 'hist' in locals() else 'N/A'}, "
                  f"target={target.shape if 'target' in locals() else 'N/A'}")
            raise
        
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
