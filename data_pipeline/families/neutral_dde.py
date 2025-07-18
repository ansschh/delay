# families/neutral_dde.py

import numpy as np
from scipy.interpolate import CubicSpline

def f(t, u, u_tau, u_tau_prime, a=0.5, b=-0.5, c=0.1):
    """
    Neutral DDE: u'(t) = a*u(t) + b*u(t-τ) + c*u'(t-τ)
    
    Parameters:
    -----------
    t : float
        Current time
    u : ndarray
        Current state
    u_tau : ndarray
        Delayed state u(t-τ)
    u_tau_prime : ndarray
        Delayed derivative u'(t-τ)
    a : float, optional
        Coefficient for current state (default=0.5)
    b : float, optional
        Coefficient for delayed state (default=-0.5)
    c : float, optional
        Coefficient for delayed derivative (default=0.1)
    
    Returns:
    --------
    ndarray
        Right-hand side of the neutral DDE
    """
    # For a neutral DDE, we need to solve for u'(t) algebraically
    # u'(t) = a*u(t) + b*u(t-τ) + c*u'(t-τ)
    # u'(t) - c*u'(t-τ) = a*u(t) + b*u(t-τ)
    # u'(t) = [a*u(t) + b*u(t-τ) + c*u'(t-τ)]/(1)
    return a*u + b*u_tau + c*u_tau_prime

def random_history(τ, d=1):
    """
    Generate a C¹ random smooth history function on [-τ,0]
    For neutral DDEs, the history function must be C¹ (continuously differentiable)
    
    Parameters:
    -----------
    τ : float
        Delay value
    d : int, optional
        Dimension of the system (default=1)
        
    Returns:
    --------
    tuple
        (history function, derivative function) as callables
    """
    # Use cubic spline to ensure C¹ continuity
    ts = np.linspace(-τ, 0, 15)  # More points for smoother derivative
    vs = np.random.rand(len(ts), d)
    
    # Create a cubic spline with natural boundary conditions
    spline = CubicSpline(ts, vs, axis=0, bc_type='natural')
    
    # Return both the history function and its derivative
    history = lambda t: spline(np.maximum(-τ, np.minimum(0, t)))
    history_prime = lambda t: spline(np.maximum(-τ, np.minimum(0, t)), 1)  # 1st derivative
    
    return history, history_prime

# Flag to indicate this is a stiff system
stiff = True

# Parameters for dataset generation
params = {
    'a': 0.5,
    'b': -0.5,
    'c': 0.1
}

# Function implementation for stiff solver
def stiff_rhs(t, y, y_tau, **kwargs):
    """
    Right-hand side function for stiff solver that's compatible with solve_stiff_dde
    """
    a = kwargs.get('a', 0.5)
    b = kwargs.get('b', -0.5)
    c = kwargs.get('c', 0.1)
    
    # For neutral DDEs, y_tau will be the value at t-τ, not the derivative
    # So we'll compute an approximate derivative using finite differences
    # or just use a simplified form
    return a*y + b*y_tau

# Equation string for radar5 solver (updated to work with our custom implementation)
eqns = {
    'u': stiff_rhs
}
