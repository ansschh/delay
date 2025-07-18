import torch
import numpy as np
from scipy import interpolate
from typing import List, Dict, Any

def pad_tensor(x, pad_size, dim=-1, mode='constant', value=0):
    """
    Pad tensor along specified dimension
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor
    pad_size : int or tuple
        Size to pad to or padding amount
    dim : int
        Dimension to pad along
    mode : str
        Padding mode
    value : float
        Constant value for padding
        
    Returns:
    --------
    torch.Tensor: Padded tensor
    """
    if isinstance(pad_size, int):
        # Pad to size
        current_size = x.shape[dim]
        if current_size >= pad_size:
            return x  # No padding needed
        padding_amount = pad_size - current_size
    else:
        # Use provided padding amount
        padding_amount = pad_size
    
    padding = [0] * (2 * x.dim())
    padding[2 * dim + 1] = padding_amount
    return torch.nn.functional.pad(x, tuple(padding), mode=mode, value=value)

def interpolate_grid(values, grid_in, grid_out, kind='linear'):
    """
    Interpolate values from one grid to another
    
    Parameters:
    -----------
    values : numpy.ndarray
        Values on the input grid
    grid_in : numpy.ndarray
        Input grid points
    grid_out : numpy.ndarray
        Output grid points
    kind : str
        Interpolation method
        
    Returns:
    --------
    numpy.ndarray: Interpolated values on output grid
    """
    f = interpolate.interp1d(grid_in, values, axis=0, kind=kind, 
                            bounds_error=False, fill_value='extrapolate')
    return f(grid_out)

def create_uniform_grid(start, end, n):
    """
    Create uniform grid from start to end with n points
    
    Parameters:
    -----------
    start : float
        Start point
    end : float
        End point
    n : int
        Number of points
        
    Returns:
    --------
    numpy.ndarray: Uniform grid
    """
    return np.linspace(start, end, n)

def to_numpy(tensor):
    """
    Convert PyTorch tensor to numpy array
    
    Parameters:
    -----------
    tensor : torch.Tensor
        PyTorch tensor
        
    Returns:
    --------
    numpy.ndarray: NumPy array
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def to_torch(array, device=None):
    """
    Convert numpy array to PyTorch tensor
    
    Parameters:
    -----------
    array : numpy.ndarray
        NumPy array
    device : torch.device
        PyTorch device
        
    Returns:
    --------
    torch.Tensor: PyTorch tensor
    """
    if isinstance(array, torch.Tensor):
        return array.to(device) if device is not None else array
    tensor = torch.from_numpy(array)
    return tensor.to(device) if device is not None else tensor

def pad_collate(batch: List[Dict[str, Any]]):
    """Custom collate_fn that stacks fixed-size tensors.

    Assumes each item in the batch is a dict with keys:
        hist:  (nx, S) tensor
        y:     (nx, T_out) tensor (already padded/truncated in Dataset)
        tau:   (1,) tensor
    """
    hist = torch.stack([item["hist"] for item in batch], dim=0)   # (B,nx,S)
    y    = torch.stack([item["y"] for item in batch], dim=0)      # (B,nx,T_out)
    tau  = torch.stack([item["tau"] for item in batch], dim=0)    # (B,1)
    return {"hist": hist, "y": y, "tau": tau}


class Timer:
    """Simple timer context manager for profiling"""
    def __init__(self, name=None):
        self.name = name
        
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self
        
    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        elapsed = self.start.elapsed_time(self.end) / 1000.0  # convert to seconds
        print(f"{self.name}: {elapsed:.4f} seconds")
