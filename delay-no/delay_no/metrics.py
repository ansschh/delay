import torch
import numpy as np
import time
from functools import wraps

def timeit(func):
    """Decorator to measure execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def l2_relative_error(pred, target):
    """
    Compute relative L2 error between prediction and target
    
    Parameters:
    -----------
    pred : torch.Tensor or numpy.ndarray
        Prediction tensor
    target : torch.Tensor or numpy.ndarray
        Target tensor
        
    Returns:
    --------
    float: Relative L2 error
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    error = np.sqrt(np.mean((pred - target)**2))
    norm = np.sqrt(np.mean(target**2))
    return error / (norm + 1e-8)

def spectral_l2_error(pred, target):
    """
    Compute L2 error in Fourier space between prediction and target
    
    Parameters:
    -----------
    pred : torch.Tensor
        Prediction tensor
    target : torch.Tensor
        Target tensor
        
    Returns:
    --------
    float: Spectral L2 error
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
        
    # Compute FFT
    pred_fft = torch.fft.rfftn(pred, dim=(-2, -1) if pred.dim() > 2 else -1)
    target_fft = torch.fft.rfftn(target, dim=(-2, -1) if target.dim() > 2 else -1)
    
    # Compute relative error in Fourier space
    error = torch.sqrt(torch.mean((pred_fft.abs() - target_fft.abs())**2))
    norm = torch.sqrt(torch.mean(target_fft.abs()**2))
    return (error / (norm + 1e-8)).item()

def energy_ratio(pred_trajectory, initial_value):
    """
    Compute energy growth ratio between final prediction and initial value
    
    Parameters:
    -----------
    pred_trajectory : torch.Tensor or numpy.ndarray
        Prediction trajectory tensor, last dim is time
    initial_value : torch.Tensor or numpy.ndarray
        Initial value tensor
        
    Returns:
    --------
    float: Energy growth ratio
    """
    if isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = pred_trajectory.detach().cpu().numpy()
    if isinstance(initial_value, torch.Tensor):
        initial_value = initial_value.detach().cpu().numpy()
        
    final_pred = pred_trajectory[..., -1]
    initial_energy = np.mean(initial_value**2)
    final_energy = np.mean(final_pred**2)
    
    return final_energy / (initial_energy + 1e-8)

def wall_clock(model, dataset, runs=100, batch_size=1, device="cuda"):
    """
    Measure wall-clock time for inference
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    dataset : torch.utils.data.Dataset
        Dataset for inference
    runs : int
        Number of runs for timing
    batch_size : int
        Batch size for inference
    device : str
        Device for computation
        
    Returns:
    --------
    float: Trajectories processed per second
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Get a sample batch
    if hasattr(dataset, '__getitem__'):
        # Create a small dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))
    else:
        # Use the provided batch directly
        batch = dataset
    
    # Ensure batch is on the correct device
    if isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            if isinstance(batch, dict):
                _ = model(**batch)
            else:
                _ = model(*batch)
    
    # Measure time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(runs):
            if isinstance(batch, dict):
                _ = model(**batch)
            else:
                _ = model(*batch)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    # Calculate throughput
    elapsed_time = end_time - start_time
    trajectories_per_second = (runs * batch_size) / elapsed_time
    
    return trajectories_per_second
