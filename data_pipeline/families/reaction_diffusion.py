# families/reaction_diffusion.py

import numpy as np
from scipy.sparse import diags
from scipy.interpolate import CubicSpline

def f(t, u, u_tau, D=0.01, a=1.0, b=1.0, nx=20):
    """
    Reaction-diffusion with delay: 
    u_t = D*∇²u + a*u(t) + b*u(t-τ)
    
    Parameters:
    -----------
    t : float
        Current time
    u : ndarray
        Current state, discretized spatial field
    u_tau : ndarray
        Delayed state u(t-τ)
    D : float, optional
        Diffusion coefficient (default=0.01)
    a : float, optional
        Coefficient for current state (default=1.0)
    b : float, optional
        Coefficient for delayed state (default=1.0)
    nx : int, optional
        Number of spatial grid points (default=20)
    
    Returns:
    --------
    ndarray
        Right-hand side of the spatially discretized DDE
    """
    # Method of lines with centered finite difference for Laplacian
    # Assuming 1D spatial domain with periodic boundary conditions
    dx = 1.0 / nx
    
    # Construct sparse finite difference matrix for Laplacian
    # For periodic boundary: u[0] connected to u[nx-1]
    diagonals = [np.ones(nx-1), -2*np.ones(nx), np.ones(nx-1)]
    offsets = [-1, 0, 1]
    laplacian = diags(diagonals, offsets, shape=(nx, nx), format='csr')
    # Fix periodic boundary
    laplacian[0, nx-1] = 1
    laplacian[nx-1, 0] = 1
    
    laplacian = laplacian / (dx**2)
    
    # Compute right-hand side
    diffusion = D * laplacian.dot(u)
    reaction = a * u + b * u_tau
    
    return diffusion + reaction

def random_history(τ, nx=20):
    """
    Generate a random smooth spatial-temporal history function on [-τ,0] x [0,1]
    
    Parameters:
    -----------
    τ : float
        Delay value
    nx : int, optional
        Number of spatial grid points (default=20)
        
    Returns:
    --------
    callable
        History function h(t) for t in [-τ,0]
    """
    # Create a space-time field with random initial values
    nt = 15  # Number of time points in history
    times = np.linspace(-τ, 0, nt)
    
    # Create random spatial profiles at each time point
    # Ensure smoothness in both space and time
    profiles = np.zeros((nt, nx))
    
    # Generate random Fourier modes for smooth spatial profiles
    n_modes = 5
    for i in range(nt):
        # Create a smooth spatial profile using a few Fourier modes
        profile = np.zeros(nx)
        for k in range(1, n_modes+1):
            a_k = np.random.randn()
            b_k = np.random.randn()
            x = np.linspace(0, 1, nx, endpoint=False)
            profile += a_k * np.sin(2*np.pi*k*x) + b_k * np.cos(2*np.pi*k*x)
        
        # Scale to [0,1] range
        profile = 0.5 + 0.4 * profile / np.max(np.abs(profile))
        profiles[i, :] = profile
    
    # Create temporal interpolation for each spatial point
    interpolators = []
    for i in range(nx):
        values = profiles[:, i]
        spline = CubicSpline(times, values)
        interpolators.append(spline)
    
    # Return a function that interpolates in time for each spatial point
    def history(t):
        t_clipped = np.maximum(-τ, np.minimum(0, t))
        result = np.zeros(nx)
        for i in range(nx):
            result[i] = interpolators[i](t_clipped)
        return result
    
    return history

# Flag to indicate this is a stiff system
stiff = True

# Parameters for dataset generation
params = {
    'D': 0.01,
    'a': 1.0,
    'b': 1.0,
    'nx': 20
}

# Equation string for radar5 solver (simplified, actual implementation would need vector form)
eqns = {
    'u': 'D*laplacian(u) + a*u + b*delay(u, tau)'
}
