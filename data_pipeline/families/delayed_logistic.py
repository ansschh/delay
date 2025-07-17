# families/delayed_logistic.py

import numpy as np
from scipy.interpolate import CubicSpline

def f(t, u, u_tau, r=2.0, K=1.0):
    """
    Delayed logistic equation: u' = r*u*(1 - u_tau/K)
    
    Parameters:
    -----------
    t : float
        Current time
    u : ndarray
        Current state
    u_tau : ndarray
        Delayed state u(t-τ)
    r : float, optional
        Growth rate (default=2.0)
    K : float, optional
        Carrying capacity (default=1.0)
    
    Returns:
    --------
    ndarray
        Right-hand side of the DDE
    """
    return r * u * (1 - u_tau / K)

def random_history(τ, d=1):
    """
    Generate a random smooth history function on [-τ,0]
    
    Parameters:
    -----------
    τ : float
        Delay value
    d : int, optional
        Dimension of the system (default=1)
        
    Returns:
    --------
    callable
        History function h(t) for t in [-τ,0]
    """
    # Random smooth cubic spline on [-τ,0]
    ts = np.linspace(-τ, 0, 10)
    # For logistic equation, ensure values are positive
    vs = 0.1 + 0.5 * np.random.rand(len(ts), d)  
    spline = CubicSpline(ts, vs, axis=0)
    return lambda t: spline(np.maximum(-τ, np.minimum(0, t)))

# Parameters for dataset generation
params = {
    'r': 2.0,
    'K': 1.0
}

# Equation string for radar5 solver
eqns = {
    'u': 'r*u*(1 - delay(u, tau)/K)'
}
