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
        # We add 1 for the temporal coordinate channel
        self.lift = ChannelMLP(in_ch + 1, hidden, hidden)  # +1 for temporal coord
        
        # Create FNO blocks for 1D temporal sequences
        self.fno_blocks = nn.ModuleList()
        
        # Ensure n_modes is properly formatted for 1D and convert from Hydra config
        # Convert from Hydra ListConfig to regular Python list/tuple
        if hasattr(n_modes, '_content'):
            n_modes = list(n_modes)
        
        if isinstance(n_modes, (list, tuple)) and len(n_modes) > 1:
            # If 2D modes provided, use only the first for 1D
            n_modes_1d = tuple([n_modes[0]])
            print(f"Converting 2D modes {n_modes} to 1D modes {n_modes_1d}")
        elif isinstance(n_modes, (list, tuple)):
            n_modes_1d = tuple(n_modes)
        else:
            n_modes_1d = tuple([n_modes])
            
        for _ in range(L):
            # Create 1D spectral convolution layer with complex dtype support
            spectral_conv = SpectralConvND(
                in_ch=hidden, 
                out_ch=hidden, 
                n_modes=n_modes_1d  # Dimensionality inferred from n_modes length
            )
            
            # Note: proj parameter is already properly initialized in SpectralConvND constructor
            # Complex dtype conversion is handled in compl_weight() method
            
            # Create FNO block
            fno_block = nn.Sequential(
                spectral_conv,
                nn.GELU(),
                ChannelMLP(hidden, hidden, hidden)
            )
            
            self.fno_blocks.append(fno_block)
        
        self.proj = ChannelMLP(hidden, hidden, out_ch)

    def add_coords(self, h):
        """Add normalized coordinate channels to input for 1D temporal sequences"""
        B, C, S = h.shape
        # Add temporal coordinate channel (normalized from -1 to 0 for history)
        s = torch.linspace(-1, 0, S, device=h.device)[None, None, :].expand(B, 1, S)
        return torch.cat([h, s], dim=1)

    def forward(self, hist):
        """
        Forward pass through the model
        
        Parameters:
        -----------
        hist : torch.Tensor
            History tensor of shape (B, C, S) for 1D temporal sequences
            
        Returns:
        --------
        torch.Tensor: Predicted next window of shape (B, C, S)
        """
        # Get actual input shape and adapt the model if necessary
        B, C, S = hist.shape
        
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
                # Recreate lift layer with correct input channel count (C + 1 for temporal coordinate)
                # Properly register the new module and handle device/dtype
                new_lift = ChannelMLP(C + 1, self.hidden, self.hidden)
                new_lift = new_lift.to(device=hist.device, dtype=hist.dtype)
                # Properly register the new module
                self.lift = new_lift
                print(f"Adjusted lift layer input channels from {old_in_ch} to {C}")
            self._first_forward = False
        
        # Add coordinate channels
        z = self.add_coords(hist)  # (B, C+1, S)
        
        # Permute to move channels to last dimension for MLP
        z_perm = z.permute(0, 2, 1)  # (B, S, C+1)
        
        # Lift to hidden dimension
        z = self.lift(z_perm)  # (B, S, hidden)
        
        # Move channels back to second dim for convolution
        z = z.permute(0, 2, 1)  # (B, hidden, S)
        
        # Apply FNO blocks with checkpointing if available
        for blk in self.fno_blocks:
            if self.training and hasattr(torch.utils, 'checkpoint'):
                z = torch.utils.checkpoint.checkpoint(blk, z)
            else:
                z = blk(z)
        
        # Project back to output dimension
        out = self.proj(z.permute(0, 2, 1)).permute(0, 2, 1)  # (B, C, S)
        
        return out
    
    def training_step(self, batch, batch_idx):
        """Lightning training step with improved shape handling for 1D sequences"""
        try:
            # Get data and print shapes (on first batch)
            hist = batch["hist"]  # Expected: (B, S, nx) from pad_collate
            target = batch["y"]   # Expected: (B, T, nx) from pad_collate
            
            if batch_idx == 0:
                print(f"[training_step] Input shapes: hist={hist.shape}, target={target.shape}")
            
            # Reshape to (B, C, S) format expected by forward pass for 1D sequences
            # Move channels (nx) to second dimension
            hist = hist.permute(0, 2, 1)  # (B, nx, S)
            target = target.permute(0, 2, 1)  # (B, nx, T)
                
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
        """Lightning validation step with improved shape handling for 1D sequences"""
        try:
            # Get data from batch
            hist = batch["hist"]  # Expected: (B, S, nx) from pad_collate
            target = batch["y"]   # Expected: (B, T, nx) from pad_collate
            
            if batch_idx == 0:
                print(f"[validation_step] Input shapes: hist={hist.shape}, target={target.shape}")
            
            # Reshape to (B, C, S) format expected by forward pass for 1D sequences
            # Move channels (nx) to second dimension
            hist = hist.permute(0, 2, 1)  # (B, nx, S)
            target = target.permute(0, 2, 1)  # (B, nx, T)
                
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
