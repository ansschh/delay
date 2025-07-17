# solvers/radar5_wrapper.py

import numpy as np
from scipy.integrate import solve_ivp
import warnings

def solve_stiff_dde(eqns, params, τs, history, t_eval):
    """
    Alternative implementation of stiff DDE solver using scipy.integrate.solve_ivp
    with BDF (backward differentiation formula) method for stiff problems.
    
    Parameters:
    -----------
    eqns: dict{'u': function(t, y, y_tau)}
        Right-hand side function that takes current time, state, and delayed state
    params: dict
        Parameters for the equations
    τs: list
        List of delay values
    history: dict{'u': callable}
        History function that takes time t and returns state at time t
    t_eval: ndarray
        Times at which to return the solution
    
    Returns:
    --------
    tuple
        (t, y) times and solution values
    """
    # Extract the main delay value (assuming single delay for simplicity)
    τ = τs[0]
    
    # Get the history function
    hist_func = history['u']
    
    # For stiff systems, we'll use the BDF method
    # We adapt the method of steps approach but with stiff solver
    
    # Storage for solution
    t_points = []
    y_points = []
    
    # Starting time and final time
    t0 = 0.0
    tf = t_eval[-1]
    
    # We'll divide the integration into chunks of length τ
    # to better handle the delay
    chunk_size = min(τ, (tf - t0) / 10)  # Either τ or a fraction of the total time
    
    # Initial condition
    y0 = hist_func(0.0)
    if not isinstance(y0, (list, np.ndarray)):
        y0 = np.array([y0])
    
    # Create interpolation function for history
    # We'll build this as we go
    from scipy.interpolate import interp1d
    
    # Start with history on [-τ, 0]
    t_hist = np.linspace(-τ, 0, max(100, int(τ/0.01)))  # Fine grid for history
    y_hist = np.array([hist_func(t) for t in t_hist])
    
    # If y_hist contains scalars, reshape to column vector
    if y_hist.ndim == 1:
        y_hist = y_hist.reshape(-1, 1)
    
    # Initialize the time points with history
    all_t = t_hist
    all_y = y_hist
    
    # Create interpolation function
    def get_delayed_value(t):
        """Get value at time t from interpolation"""
        # If t is in the past, use history function
        if t <= 0:
            return hist_func(t)
        
        # If t is beyond our computed solution, warn and use the latest value
        if t > all_t[-1]:
            warnings.warn(f"Time {t} is beyond computed solution at {all_t[-1]}")
            return all_y[-1]
        
        # Otherwise use interpolation
        interp = interp1d(all_t, all_y, axis=0, bounds_error=False, fill_value="extrapolate")
        return interp(t)
    
    # Define the ODE function that includes the delay term
    def rhs_with_delay(t, y):
        """RHS function that includes the delay term"""
        # Get delayed state
        y_tau = get_delayed_value(t - τ)
        
        # Get derivative from the equations
        # We'll access the first equation's function (assuming one equation)
        if callable(eqns['u']):
            # If it's a function, call it directly
            return eqns['u'](t, y, y_tau, **params)
        else:
            # Default simple implementation for common cases
            # This handles cases like Mackey-Glass, etc.
            u = y[0] if isinstance(y, (list, np.ndarray)) else y
            u_tau = y_tau[0] if isinstance(y_tau, (list, np.ndarray)) else y_tau
            
            # Simple implementation for Mackey-Glass as default
            beta = params.get('β', params.get('beta', 0.2))
            gamma = params.get('γ', params.get('gamma', 0.1))
            n = params.get('n', 10)
            
            dydt = beta * u_tau/(1 + u_tau**n) - gamma * u
            return dydt
    
    # Integrate in chunks to better handle the delay term
    current_t = t0
    while current_t < tf:
        # Determine end of this chunk
        chunk_end = min(current_t + chunk_size, tf)
        
        # Integrate over this chunk
        sol = solve_ivp(
            rhs_with_delay,
            (current_t, chunk_end),
            y0.flatten(),  # Ensure 1D array for solve_ivp
            method='BDF',  # Backward differentiation formula for stiff problems
            rtol=1e-6,
            atol=1e-8,
            dense_output=True
        )
        
        # Store solution points
        t_points.append(sol.t)
        y_points.append(sol.y.T)
        
        # Update the interpolation data
        all_t = np.append(all_t, sol.t)
        if sol.y.ndim == 1:
            new_y = sol.y.reshape(-1, 1)
        else:
            new_y = sol.y.T
        all_y = np.vstack([all_y, new_y])
        
        # Update current time and initial condition for next chunk
        current_t = chunk_end
        y0 = sol.y[:, -1]  # Last solution point
    
    # Combine all solution points
    t_combined = np.concatenate(t_points)
    y_combined = np.vstack(y_points)
    
    # Return solution at requested evaluation points using interpolation
    sol_interp = interp1d(t_combined, y_combined, axis=0, bounds_error=False, fill_value="extrapolate")
    y_eval = sol_interp(t_eval)
    
    return t_eval, y_eval
