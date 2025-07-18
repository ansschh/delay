import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..layers.spectral_conv_nd import SpectralConvND

class LearnableKernel(nn.Module):
    """
    Learnable memory kernel for the integral formulation
    """
    def __init__(self, S=32, in_ch=1, out_ch=1):
        super().__init__()
        self.S = S
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Complex-valued weights initialized randomly
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, S, dtype=torch.cfloat))
        
    def forward(self, hist):
        """
        Apply learnable kernel to history tensor
        
        Parameters:
        -----------
        hist : torch.Tensor
            History tensor of shape (B, I, S)
            
        Returns:
        --------
        torch.Tensor: Convolution result at current time of shape (B, O)
        """
        # Apply convolution in Fourier space
        hist_ft = torch.fft.rfft(hist, dim=-1)
        w_ft = self.weight[..., :hist_ft.size(-1)]  # match spectrum
        conv_ft = (hist_ft.unsqueeze(1) * w_ft).sum(2)  # (B, O, Freq)
        conv = torch.fft.irfft(conv_ft, n=hist.size(-1), dim=-1)
        
        # Return value at current time step (end of history)
        return conv[..., -1]  # (B, O)


class LocalFunction(nn.Module):
    """
    Local function component for the integral approach
    """
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )
        
    def forward(self, u):
        """
        Apply local function to current state
        
        Parameters:
        -----------
        u : torch.Tensor
            Current state tensor of shape (B, D)
            
        Returns:
        --------
        torch.Tensor: Local function output of shape (B, D)
        """
        return self.mlp(u)


def euler_integrate(u0, f_loc, kernel, hist, Δt, N):
    """
    Euler integration of the memory kernel model
    
    Parameters:
    -----------
    u0 : torch.Tensor
        Initial condition of shape (B, D)
    f_loc : callable
        Local function module
    kernel : callable
        Memory kernel module
    hist : torch.Tensor
        History tensor of shape (B, D, S)
    Δt : float
        Time step
    N : int
        Number of steps to integrate
        
    Returns:
    --------
    tuple:
        - Trajectory tensor of shape (B, N+1, D)
        - Updated history tensor of shape (B, D, S)
    """
    us = [u0]
    for _ in range(N):
        # Compute derivative
        conv = kernel(hist)  # (B, D)
        du = f_loc(u0) + conv
        
        # Euler step
        u1 = u0 + Δt * du
        
        # Update history
        hist = torch.cat([hist[..., 1:], u1.unsqueeze(-1)], dim=-1)
        u0 = u1
        us.append(u1)
        
    return torch.stack(us, dim=1), hist


class SpatialExtension(nn.Module):
    """
    Optional spatial extension for 2D problems (reaction-diffusion)
    """
    def __init__(self, in_ch, hidden_ch=64, n_modes=(16, 16)):
        super().__init__()
        self.fno = nn.Sequential(
            SpectralConvND(in_ch, hidden_ch, n_modes),
            nn.GELU(),
            SpectralConvND(hidden_ch, in_ch, n_modes)
        )
        
    def forward(self, u):
        """
        Apply spatial operator to input field
        
        Parameters:
        -----------
        u : torch.Tensor
            Input field of shape (B, C, H, W)
            
        Returns:
        --------
        torch.Tensor: Output field of same shape
        """
        return self.fno(u)


class KernelLit(pl.LightningModule):
    """
    Lightning module for Memory-kernel integral approach (Variant C)
    """
    def __init__(self, S=32, in_ch=1, out_ch=1, hidden=64, n_spatial_modes=None,
                 lr=3e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.S = S
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Model components
        self.kernel = LearnableKernel(S=S, in_ch=in_ch, out_ch=out_ch)
        self.local_fn = LocalFunction(in_ch, hidden_dim=hidden)
        
        # Optional spatial extension for 2D problems
        if n_spatial_modes is not None:
            self.spatial = SpatialExtension(in_ch, hidden, n_spatial_modes)
        else:
            self.spatial = None
    
    def forward(self, hist, u0, dt, steps=10):
        """
        Forward pass with integration
        
        Parameters:
        -----------
        hist : torch.Tensor
            History tensor of shape (B, D, S)
        u0 : torch.Tensor
            Initial condition of shape (B, D)
        dt : torch.Tensor or float
            Time step
        steps : int, optional
            Number of integration steps
            
        Returns:
        --------
        torch.Tensor: Trajectory tensor of shape (B, steps+1, D)
        """
        if isinstance(dt, torch.Tensor):
            dt = dt.item()
            
        # Integration
        traj, _ = euler_integrate(u0, self.local_fn, self.kernel, hist, dt, steps)
        
        # Apply spatial extension if available
        if self.spatial is not None and len(traj.shape) > 3:
            # Reshape for spatial processing
            B, T, D = traj.shape
            traj = traj.view(B, T, -1, *traj.shape[3:])  # Assume D = C*H*W
            
            # Apply spatial operator to each time step
            spatial_out = []
            for t in range(T):
                spatial_out.append(self.spatial(traj[:, t]))
            
            traj = torch.stack(spatial_out, dim=1)
        
        return traj
    
    def training_step(self, batch, batch_idx):
        """Lightning training step"""
        hist = batch["hist"]
        u0 = batch["u0"]
        target = batch["target"]
        dt = batch["dt"].item()
        
        # Determine integration steps based on target
        steps = target.shape[1] - 1 if len(target.shape) > 2 else 1
        
        # Forward pass with integration
        traj = self.forward(hist, u0, dt, steps)
        
        # Match output format to target
        if len(target.shape) > 2:
            pred = traj[:, 1:, :]  # Skip initial condition
        else:
            pred = traj[:, -1, :]  # Take only final state
            
        # Calculate loss
        mse_loss = F.mse_loss(pred, target)
        
        self.log("train_loss", mse_loss)
        self.log("train_mse", mse_loss)
        
        return mse_loss
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step"""
        hist = batch["hist"]
        u0 = batch["u0"]
        target = batch["target"]
        dt = batch["dt"].item()
        
        # Determine integration steps based on target
        steps = target.shape[1] - 1 if len(target.shape) > 2 else 1
        
        # Forward pass with integration
        with torch.no_grad():
            traj = self.forward(hist, u0, dt, steps)
        
        # Match output format to target
        if len(target.shape) > 2:
            pred = traj[:, 1:, :]  # Skip initial condition
        else:
            pred = traj[:, -1, :]  # Take only final state
            
        # Calculate losses
        mse_loss = F.mse_loss(pred, target)
        l2_loss = torch.sqrt(mse_loss)
        
        self.log("val_mse", mse_loss)
        self.log("val_l2", l2_loss)
        
        # Check stability with longer rollout
        if batch_idx == 0:
            with torch.no_grad():
                # Roll out for longer (2*steps)
                long_traj = self.forward(hist[:2], u0[:2], dt, steps * 2)
                
                # Check energy growth
                initial_energy = torch.mean(u0[:2] ** 2)
                final_energy = torch.mean(long_traj[:, -1] ** 2)
                energy_ratio = final_energy / (initial_energy + 1e-8)
                
                self.log("rollout_energy", energy_ratio)
        
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
