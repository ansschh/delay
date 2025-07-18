import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.autograd.functional as autograd_func
from ..layers.spectral_conv_nd import SpectralConvND

class HistEncoder1DFNO(nn.Module):
    """
    1D FNO-based history encoder for Method-of-steps operator
    """
    def __init__(self, S=16, n_modes=4, hidden=64):
        super().__init__()
        self.fno1d = nn.Sequential(
            SpectralConvND(1, hidden, (n_modes,)),
            nn.GELU(),
            SpectralConvND(hidden, hidden, (n_modes,)),
            nn.GELU()
        )
        
    def forward(self, h):
        """
        Forward pass through the encoder
        
        Parameters:
        -----------
        h : torch.Tensor
            History tensor of shape (B, 1, S)
            
        Returns:
        --------
        torch.Tensor: Encoded history vector of shape (B, hidden)
        """
        z = self.fno1d(h)  # (B, hidden, S)
        return z.mean(-1)  # global avg pooling -> (B, hidden)


class StepOperator(nn.Module):
    """
    Step operator for Method-of-steps approach
    Maps history encoding and delay to the next time window
    """
    def __init__(self, hist_dim, tau_dim=16, out_len=32):
        super().__init__()
        self.tau_embed = nn.Sequential(
            nn.Linear(1, tau_dim), 
            nn.GELU(), 
            nn.Linear(tau_dim, tau_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(hist_dim + tau_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_len)
        )
        
    def forward(self, hist_vec, tau):
        """
        Forward pass through the step operator
        
        Parameters:
        -----------
        hist_vec : torch.Tensor
            Encoded history vector of shape (B, hist_dim)
        tau : torch.Tensor
            Delay parameter of shape (B, 1)
            
        Returns:
        --------
        torch.Tensor: Predicted next time window of shape (B, out_len)
        """
        tau_vec = self.tau_embed(tau)
        z = torch.cat([hist_vec, tau_vec], dim=-1)
        return self.mlp(z)


def rollout(hist_enc, step_op, h0, tau, K, Δt_idx):
    """
    Roll out prediction for K steps
    
    Parameters:
    -----------
    hist_enc : nn.Module
        History encoder network
    step_op : nn.Module
        Step operator network
    h0 : torch.Tensor
        Initial history of shape (B, 1, S)
    tau : torch.Tensor
        Delay parameter of shape (B, 1)
    K : int
        Number of steps to roll out
    Δt_idx : int
        Index to shift history by for each step
        
    Returns:
    --------
    torch.Tensor: Predictions for K steps of shape (B, K, out_len)
    """
    h = h0  # (B, 1, S)
    preds = []
    
    for k in range(K):
        vec = hist_enc(h)  # (B, hidden)
        y = step_op(vec, tau)  # (B, out_len)
        preds.append(y)
        
        # Shift history and append new prediction
        h = torch.cat([h[:, :, Δt_idx:], y.unsqueeze(1)], dim=-1)
        
    return torch.stack(preds, dim=1)  # (B, K, out_len)


class StepsLit(pl.LightningModule):
    """
    Lightning module for Method-of-steps approach (Variant B)
    """
    def __init__(self, hist_dim=64, tau_dim=16, out_len=32, S=16, n_modes=4, 
                 K=5, Δt_idx=None, spectral_penalty=0.1, lr=3e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.hist_dim = hist_dim
        self.tau_dim = tau_dim
        self.out_len = out_len
        self.S = S
        self.n_modes = n_modes
        self.K = K
        self.Δt_idx = Δt_idx or S // 4  # Default: shift by 1/4 of history length
        self.spectral_penalty = spectral_penalty
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Model components
        self.hist_enc = HistEncoder1DFNO(S=S, n_modes=n_modes, hidden=hist_dim)
        self.step_op = StepOperator(hist_dim, tau_dim, out_len)
        
        # Step counter for spectral radius penalty
        self.global_step_counter = 0
        
    def forward(self, h0, tau, K=None):
        """
        Forward pass with rollout
        
        Parameters:
        -----------
        h0 : torch.Tensor
            Initial history of shape (B, 1, S)
        tau : torch.Tensor
            Delay parameter of shape (B, 1)
        K : int, optional
            Number of steps to roll out, defaults to self.K
            
        Returns:
        --------
        torch.Tensor: Predictions for K steps
        """
        K = K or self.K
        return rollout(self.hist_enc, self.step_op, h0, tau, K, self.Δt_idx)
    
    def compute_spectral_radius(self, h, tau):
        """
        Compute the spectral radius of the Jacobian of the step operator
        
        Parameters:
        -----------
        h : torch.Tensor
            History tensor of shape (1, 1, S)
        tau : torch.Tensor
            Delay parameter of shape (1, 1)
            
        Returns:
        --------
        float: Spectral radius (largest eigenvalue)
        """
        # Encode history
        h_enc = self.hist_enc(h)
        h_enc.requires_grad_(True)
        
        # Define function for Jacobian calculation
        def step_op_fn(h_vec):
            return self.step_op(h_vec, tau)
        
        # Compute Jacobian
        try:
            J = autograd_func.jacobian(step_op_fn, h_enc)
            # Reshape to square matrix
            J = J.reshape(J.shape[0], -1)
            
            # Compute singular values and return the largest one
            s = torch.linalg.svdvals(J)
            return s[0].item()  # Spectral radius
        except Exception as e:
            print(f"Error computing spectral radius: {e}")
            return 1.0  # Default: assume stable
    
    def training_step(self, batch, batch_idx):
        """Lightning training step"""
        h0, tau, gt = batch
        
        # Forward pass with rollout
        y_pred = self.forward(h0, tau)
        
        # Calculate MSE loss
        mse_loss = F.mse_loss(y_pred, gt)
        
        # Add spectral radius penalty every 10 steps on a small batch
        spectral_penalty = 0
        self.global_step_counter += 1
        if self.spectral_penalty > 0 and self.global_step_counter % 10 == 0:
            # Use a single sample for efficiency
            sample_h = h0[:1]
            sample_tau = tau[:1]
            radius = self.compute_spectral_radius(sample_h, sample_tau)
            spectral_penalty = max(0, radius - 1.0) ** 2
            self.log("spectral_radius", radius)
            self.log("spectral_penalty", spectral_penalty)
        
        # Total loss
        loss = mse_loss + self.spectral_penalty * spectral_penalty
        
        self.log("train_loss", loss)
        self.log("train_mse", mse_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step"""
        h0, tau, gt = batch
        
        # Forward pass with rollout
        y_pred = self.forward(h0, tau)
        
        # Calculate losses
        mse_loss = F.mse_loss(y_pred, gt)
        l2_loss = torch.sqrt(mse_loss)
        
        self.log("val_mse", mse_loss)
        self.log("val_l2", l2_loss)
        
        # Check stability with longer rollout
        if batch_idx == 0:
            with torch.no_grad():
                # Roll out for longer (2*K steps)
                long_pred = self.forward(h0[:2], tau[:2], K=2*self.K)
                
                # Check energy growth
                initial_energy = torch.mean(h0[:2] ** 2)
                final_energy = torch.mean(long_pred[:, -1] ** 2)
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
