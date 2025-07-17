# utils/history_generators.py

import numpy as np
from scipy.interpolate import CubicSpline

def cubic_spline_history(τ, d=1, num_points=10, min_val=0.0, max_val=1.0):
    """
    Generate a random smooth history function using cubic splines
    
    Parameters:
    -----------
    τ : float
        Delay value
    d : int, optional
        Dimension of the system (default=1)
    num_points : int, optional
        Number of control points for the spline (default=10)
    min_val : float, optional
        Minimum value for random history (default=0.0)
    max_val : float, optional
        Maximum value for random history (default=1.0)
        
    Returns:
    --------
    callable
        History function h(t) for t in [-τ,0]
    """
    ts = np.linspace(-τ, 0, num_points)
    vs = min_val + (max_val - min_val) * np.random.rand(len(ts), d)
    spline = CubicSpline(ts, vs, axis=0)
    return lambda t: spline(np.maximum(-τ, np.minimum(0, t)))

def fourier_history(τ, d=1, num_modes=5, amplitude=0.5, offset=0.5):
    """
    Generate a random smooth history function using Fourier series
    
    Parameters:
    -----------
    τ : float
        Delay value
    d : int, optional
        Dimension of the system (default=1)
    num_modes : int, optional
        Number of Fourier modes (default=5)
    amplitude : float, optional
        Amplitude of oscillations (default=0.5)
    offset : float, optional
        Vertical offset (default=0.5)
        
    Returns:
    --------
    callable
        History function h(t) for t in [-τ,0]
    """
    # Generate random Fourier coefficients
    a_coefs = np.random.randn(num_modes, d) * amplitude / np.sqrt(num_modes)
    b_coefs = np.random.randn(num_modes, d) * amplitude / np.sqrt(num_modes)
    
    def history(t):
        # Ensure t is within bounds
        t = np.maximum(-τ, np.minimum(0, t))
        
        # Scale time to [0, 2π]
        t_scaled = 2 * np.pi * (t + τ) / τ
        
        # Start with the offset
        result = np.ones((1, d)) * offset
        
        # Add each Fourier mode
        for k in range(num_modes):
            result += a_coefs[k] * np.sin((k + 1) * t_scaled) + b_coefs[k] * np.cos((k + 1) * t_scaled)
        
        return result.flatten() if d == 1 else result
    
    return history

def filtered_brownian_history(τ, d=1, num_points=100, filter_width=5, min_val=0.0, max_val=1.0):
    """
    Generate a random history function using filtered Brownian motion
    
    Parameters:
    -----------
    τ : float
        Delay value
    d : int, optional
        Dimension of the system (default=1)
    num_points : int, optional
        Number of points for discretization (default=100)
    filter_width : int, optional
        Width of the moving average filter (default=5)
    min_val : float, optional
        Minimum value for random history (default=0.0)
    max_val : float, optional
        Maximum value for random history (default=1.0)
        
    Returns:
    --------
    callable
        History function h(t) for t in [-τ,0]
    """
    # Generate Brownian motion
    ts = np.linspace(-τ, 0, num_points)
    dW = np.random.randn(num_points, d)
    W = np.cumsum(dW, axis=0)
    
    # Apply moving average filter for smoothing
    kernel = np.ones(filter_width) / filter_width
    W_smoothed = np.zeros_like(W)
    
    for i in range(d):
        # Pad the signal for convolution
        padded = np.pad(W[:, i], (filter_width//2, filter_width//2), mode='edge')
        W_smoothed[:, i] = np.convolve(padded, kernel, mode='valid')
    
    # Scale to [min_val, max_val]
    for i in range(d):
        W_min = np.min(W_smoothed[:, i])
        W_max = np.max(W_smoothed[:, i])
        if W_max > W_min:  # Avoid division by zero
            W_smoothed[:, i] = min_val + (max_val - min_val) * (W_smoothed[:, i] - W_min) / (W_max - W_min)
        else:
            W_smoothed[:, i] = (min_val + max_val) / 2
    
    # Create interpolation function
    spline = CubicSpline(ts, W_smoothed, axis=0)
    return lambda t: spline(np.maximum(-τ, np.minimum(0, t)))

def c1_history_for_neutral_dde(τ, d=1, num_points=15):
    """
    Generate a C¹ random history function suitable for neutral DDEs
    
    Parameters:
    -----------
    τ : float
        Delay value
    d : int, optional
        Dimension of the system (default=1)
    num_points : int, optional
        Number of control points for the spline (default=15)
        
    Returns:
    --------
    tuple
        (history function, derivative function) as callables
    """
    # Use cubic spline to ensure C¹ continuity
    ts = np.linspace(-τ, 0, num_points)
    vs = np.random.rand(len(ts), d)
    
    # Create a cubic spline with natural boundary conditions
    spline = CubicSpline(ts, vs, axis=0, bc_type='natural')
    
    # Return both the history function and its derivative
    history = lambda t: spline(np.maximum(-τ, np.minimum(0, t)))
    history_prime = lambda t: spline(np.maximum(-τ, np.minimum(0, t)), 1)  # 1st derivative
    
    return history, history_prime
