# families/mackey_glass.py

import numpy as np
from scipy.interpolate import CubicSpline

def f(t, u, u_tau, β=0.2, γ=0.1, n=10):
    """
    Mackey-Glass equation: u' = β u(t-τ)/(1+u^n) - γ u
    
    Parameters:
    -----------
    t : float
        Current time
    u : ndarray
        Current state
    u_tau : ndarray
        Delayed state u(t-τ)
    β : float, optional
        Production rate parameter (default=0.2)
    γ : float, optional
        Decay rate parameter (default=0.1)
    n : float, optional
        Hill coefficient (default=10)
    
    Returns:
    --------
    ndarray
        Right-hand side of the DDE
    """
    return β * u_tau/(1 + u_tau**n) - γ * u

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
    vs = np.random.rand(len(ts), d)
    spline = CubicSpline(ts, vs, axis=0)
    return lambda t: spline(np.maximum(-τ, np.minimum(0, t)))

# Parameters for dataset generation
params = {
    'β': 0.2,
    'γ': 0.1,
    'n': 10
}

# Equation string for radar5 solver
eqns = {
    'u': 'beta * delay(u, tau)/(1 + delay(u, tau)^n) - gamma * u'
}
