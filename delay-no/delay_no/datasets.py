import torch
import numpy as np
import pickle
from scipy import interpolate
from torch.utils.data import Dataset, DataLoader
from delay_no.utils import pad_collate
import os

def load_dde_dataset(path):
    """Load a DDE dataset from a pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def interpolate_hist(hist, s_new, kind='linear'):
    """Interpolate history to new timepoints
    
    Parameters:
    -----------
    hist : array or callable
        History values or function
    s_new : array
        New time points
    kind : str
        Interpolation method
        
    Returns:
    --------
    array : Interpolated history
    """
    if callable(hist):
        # If history is a function, simply evaluate at the new points
        return np.array([hist(s) for s in s_new])
    elif isinstance(hist, dict):
        # Handle dict-based history where hist = {'t': [...], 'y': [...]} as seen in dataset files
        hist_t = np.asarray(hist.get('t'))
        hist_y = np.asarray(hist.get('y'))
        if hist_t is None or hist_y is None:
            raise ValueError("History dict must contain 't' and 'y' keys")
        if hist_t.ndim != 1:
            raise ValueError(f"hist['t'] must be 1D, got shape {hist_t.shape}")
        if hist_y.shape[0] != hist_t.shape[0]:
            raise ValueError(
                f"hist['t'] length {hist_t.shape[0]} does not match hist['y'] first dimension {hist_y.shape[0]}")
        # interpolate along time axis (first axis)
        f = interpolate.interp1d(hist_t, hist_y, axis=0, kind=kind, fill_value="extrapolate")
        return f(s_new)
    else:
        # Handle array-like history data
        try:
            # Convert hist to numpy array if it isn't already
            hist_array = np.asarray(hist)
            
            # Check if hist is 1D or multi-dimensional
            if hist_array.ndim == 1:
                # Simple 1D case
                hist_s = np.linspace(-1, 0, len(hist_array))  # Normalized time points
                f = interpolate.interp1d(hist_s, hist_array, kind=kind, fill_value="extrapolate")
                return f(s_new)
            else:
                # For multi-dimensional history (e.g., with spatial dimensions)
                # We assume the first dimension is time
                hist_s = np.linspace(-1, 0, hist_array.shape[0])  # Normalized time points
                f = interpolate.interp1d(hist_s, hist_array, axis=0, kind=kind, fill_value="extrapolate")
                return f(s_new)
                
        except Exception as e:
            # Add more debug information to help diagnose the error
            hist_shape = getattr(hist, 'shape', 'unknown shape')
            s_new_shape = getattr(s_new, 'shape', 'unknown shape')
            hist_type = type(hist)
            s_type = type(s_new)
            
            err_msg = (f"Interpolation failed with error: {str(e)}\n"
                     f"hist type: {hist_type}, shape: {hist_shape}\n"
                     f"s_new type: {s_type}, shape: {s_new_shape}")
            raise ValueError(err_msg) from e

class DDEChunk(Dataset):
    """Base dataset class for DDE samples"""
    def __init__(self, data_path, transform=None):
        self.samples = load_dde_dataset(data_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Default implementation - override in subclasses
        hist, tau, t, y = self.samples[idx]
        sample = {"hist": hist, "tau": tau, "t": t, "y": y}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class StackedHistoryDataset(DDEChunk):
    """Dataset for Stacked-history FNO approach"""
    def __init__(self, data_path, S=16, horizon=None, nx=None):
        super().__init__(data_path)
        self.S = S
        self.horizon = horizon
        self.nx = nx
        # Validate mandatory configs
        if self.horizon is None or self.nx is None:
            raise ValueError("StackedHistoryDataset requires explicit horizon and nx in the data config.")
        # Derive dt and T_out (number of prediction steps) from the first sample
        _, _, t0, y0 = self.samples[0]
        self.dt = t0[1] - t0[0] if len(t0) > 1 else 1.0
        self.T_out = int(round(self.horizon / self.dt))
        if self.T_out <= 0:
            raise ValueError(f"Invalid horizon {self.horizon} with dt {self.dt}")
        
    def __getitem__(self, idx):
        hist, tau, t, y = self.samples[idx]
        
        # Set defaults if not specified
        if self.horizon is None:
            self.horizon = t[-1] - t[0]  # Use entire time span
            
        if self.nx is None:
            # Try to infer spatial dimension
            if len(y.shape) > 1:
                self.nx = y.shape[1]
            else:
                self.nx = 1
        
        # Create uniform grid for history
        s_axis = np.linspace(-tau, 0, self.S)
        
        # Interpolate history to uniform grid
        if callable(hist):
            hist_grid = np.array([hist(s) for s in s_axis])
        else:
            # Reshape to ensure proper dimensionality
            hist_grid = interpolate_hist(hist, s_axis)
            
        # Ensure hist_grid shape (S,nx)
        if hist_grid.ndim == 1:
            hist_grid = hist_grid.reshape(self.S, 1)
        if hist_grid.shape[1] == 1 and self.nx > 1:
            # replicate single channel across nx
            hist_grid = np.repeat(hist_grid, self.nx, axis=1)
        elif hist_grid.shape[1] != self.nx:
            raise ValueError(f"hist_grid channels {hist_grid.shape[1]} != nx {self.nx}")
            
        # Build fixed-length target window (nx, T_out)
        end_time = t[0] + self.horizon
        target_mask = t <= end_time
        target = y[target_mask]
        # truncate / pad to T_out
        if target.shape[0] > self.T_out:
            target = target[:self.T_out]
        elif target.shape[0] < self.T_out:
            pad_len = self.T_out - target.shape[0]
            pad_block = np.zeros((pad_len, self.nx), dtype=target.dtype)
            target = np.concatenate([target, pad_block], axis=0)
        
        # Convert to tensors (shape nx,S and nx,T_out)
        hist_tensor = torch.tensor(hist_grid.T, dtype=torch.float32)  # (nx,S)
        target_tensor = torch.tensor(target.T, dtype=torch.float32)   # (nx,T_out)
        tau_tensor = torch.tensor([tau], dtype=torch.float32)
        
        return {
            "hist": hist_tensor.permute(1, 0) if len(hist_tensor.shape) > 1 else hist_tensor.unsqueeze(0),  # (nx, S)
            "tau": tau_tensor,
            "y": target_tensor.permute(1, 0) if len(target_tensor.shape) > 1 else target_tensor.unsqueeze(0)  # (nx, T)
        }

class MethodStepsDataset(DDEChunk):
    """Dataset for Method-of-steps approach"""
    def __init__(self, data_path, S=16, K=5, dt=0.1):
        super().__init__(data_path)
        self.S = S  # Number of history points
        self.K = K  # Number of steps to predict
        self.dt = dt  # Time step size
        
    def __getitem__(self, idx):
        hist, tau, t, y = self.samples[idx]
        
        # Create uniform grid for history
        s_axis = np.linspace(-tau, 0, self.S)
        
        # Interpolate history to uniform grid
        hist_grid = interpolate_hist(hist, s_axis)
        
        # Reshape if needed
        if len(hist_grid.shape) == 1:
            hist_grid = hist_grid.reshape(self.S, 1)
            
        # Find index corresponding to dt
        dt_indices = np.where(np.isclose(t - t[0], self.dt))[0]
        if len(dt_indices) > 0:
            dt_idx = dt_indices[0]
        else:
            # Interpolate if exact dt not found
            dt_idx = max(1, int(self.dt / (t[1] - t[0])))
            
        # Get K steps of ground truth
        gt = []
        for k in range(self.K):
            start_idx = k * dt_idx
            end_idx = min((k + 1) * dt_idx, len(y))
            if start_idx < len(y):
                step_y = y[start_idx:end_idx]
                gt.append(step_y)
            else:
                # Pad with zeros if not enough data
                gt.append(np.zeros_like(y[:dt_idx]))
                
        gt_array = np.concatenate(gt)
                
        # Convert to tensors
        hist_tensor = torch.tensor(hist_grid, dtype=torch.float32).unsqueeze(0)  # (1, S, nx)
        tau_tensor = torch.tensor([tau], dtype=torch.float32).unsqueeze(0)
        gt_tensor = torch.tensor(gt_array, dtype=torch.float32)
        
        return hist_tensor, tau_tensor, gt_tensor

class KernelIntegralDataset(DDEChunk):
    """Dataset for Memory-kernel integral approach"""
    def __init__(self, data_path, S=32, dt=0.1):
        super().__init__(data_path)
        self.S = S  # History length
        self.dt = dt  # Time step
        
    def __getitem__(self, idx):
        hist, tau, t, y = self.samples[idx]
        
        # Create uniform grid for history
        s_axis = np.linspace(-tau, 0, self.S)
        
        # Interpolate history
        hist_grid = interpolate_hist(hist, s_axis)
        
        # Reshape if needed
        if len(hist_grid.shape) == 1:
            hist_grid = hist_grid.reshape(self.S, 1)
        elif len(hist_grid.shape) > 2:
            # Handle multi-dim case (e.g., reaction-diffusion)
            hist_grid = hist_grid.reshape(self.S, -1)
            
        # Initial point
        u0 = hist_grid[-1]
        
        # Find prediction horizon
        N = int(1.0 / self.dt)  # Default: predict 1.0 time units ahead
        target_idx = min(len(y), N)
        target = y[:target_idx]
        
        # Convert to tensors
        hist_tensor = torch.tensor(hist_grid, dtype=torch.float32).permute(1, 0)  # (nx, S)
        u0_tensor = torch.tensor(u0, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        if len(target_tensor.shape) == 1:
            target_tensor = target_tensor.unsqueeze(1)
            
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)
        
        return {
            "hist": hist_tensor,
            "u0": u0_tensor,
            "target": target_tensor,
            "dt": dt_tensor,
            "tau": torch.tensor([tau], dtype=torch.float32)
        }
        
def create_data_module(variant, data_config):
    """Factory function for creating dataset and dataloader based on variant"""
    family = data_config.get("family", "mackey_glass")
    data_dir = data_config.get("data_dir", "data")
    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    
    # Build paths
    if family == "all":
        train_paths = [
            os.path.join(data_dir, "combined", f"{f}_train.pkl") 
            for f in ["mackey_glass", "delayed_logistic", "neutral_dde", "reaction_diffusion"]
        ]
        test_paths = [
            os.path.join(data_dir, "combined", f"{f}_test.pkl")
            for f in ["mackey_glass", "delayed_logistic", "neutral_dde", "reaction_diffusion"]
        ]
    else:
        train_path = os.path.join(data_dir, family, f"{family}.pkl")
        test_path = os.path.join(data_dir, family, f"{family}.pkl")
        
        # Fallback to combined directory if not found in individual directories
        if not os.path.exists(train_path):
            train_path = os.path.join(data_dir, "combined", f"{family}_train.pkl")
            test_path = os.path.join(data_dir, "combined", f"{family}_test.pkl")
            
        train_paths = [train_path]
        test_paths = [test_path]
    
    # Create appropriate dataset
    if variant == "stacked":
        S = data_config.get("S", 16)
        train_datasets = [StackedHistoryDataset(path, S=S) for path in train_paths if os.path.exists(path)]
        test_datasets = [StackedHistoryDataset(path, S=S) for path in test_paths if os.path.exists(path)]
    elif variant == "steps":
        S = data_config.get("S", 16)
        K = data_config.get("K", 5)
        dt = data_config.get("dt", 0.1)
        train_datasets = [MethodStepsDataset(path, S=S, K=K, dt=dt) for path in train_paths if os.path.exists(path)]
        test_datasets = [MethodStepsDataset(path, S=S, K=K, dt=dt) for path in test_paths if os.path.exists(path)]
    elif variant == "kernel":
        S = data_config.get("S", 32)
        dt = data_config.get("dt", 0.1)
        train_datasets = [KernelIntegralDataset(path, S=S, dt=dt) for path in train_paths if os.path.exists(path)]
        test_datasets = [KernelIntegralDataset(path, S=S, dt=dt) for path in test_paths if os.path.exists(path)]
    else:
        raise ValueError(f"Unknown variant: {variant}")
        
    # Combine datasets if multiple
    if len(train_datasets) > 1:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)
    else:
        train_dataset = train_datasets[0] if train_datasets else None
        test_dataset = test_datasets[0] if test_datasets else None
        
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate
    ) if train_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate
    ) if test_dataset else None
    
    return train_loader, test_loader

def create_dataloaders(model_config, data_config):
    """Function to create dataloaders for training and testing
    
    This is the function imported by train.py and evaluate.py
    
    Parameters:
    -----------
    model_config : dict
        Model configuration with variant information
    data_config : dict
        Data configuration with dataset information
        
    Returns:
    --------
    tuple: (train_loader, test_loader)
    """
    # Extract model variant from config
    variant = model_config.get("variant", "stacked")
    
    # Call the underlying implementation
    return create_data_module(variant, data_config)
