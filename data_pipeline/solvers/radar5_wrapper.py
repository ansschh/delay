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
    history: dict{'u': callable or tuple}
        History function that takes time t and returns state at time t,
        or a tuple of (history_func, history_prime_func) for neutral DDEs
    t_eval: ndarray
        Times at which to return the solution
    
    Returns:
    --------
    tuple
        (t, y) times and solution values
    """
    # Extract the main delay value (assuming single delay for simplicity)
    τ = τs[0]
    
    # Get the history function - handle both regular and neutral DDEs
    hist_item = history['u']
    
    # Check if we have a neutral DDE (history returns a tuple of functions)
    is_neutral = isinstance(hist_item, tuple) and len(hist_item) == 2 and all(callable(f) for f in hist_item)
    
    if is_neutral:
        # For neutral DDEs: hist_func is the first element of the tuple
        hist_func = hist_item[0]
        hist_prime_func = hist_item[1]  # Save the derivative function for later use
    else:
        # For regular DDEs
        hist_func = hist_item
        hist_prime_func = None
    
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
        
    # Make sure y0 is a flattened array as required by solve_ivp
    if hasattr(y0, 'flatten'):
        y0 = y0.flatten()
    else:
        y0 = np.array([y0]).flatten()
    
    # Create interpolation function for history
    # We'll build this as we go
    from scipy.interpolate import interp1d
    
    # Start with history on [-τ, 0]
    t_hist = np.linspace(-τ, 0, max(100, int(τ/0.01)))  # Fine grid for history
    
    try:
        y_hist = np.array([hist_func(t) for t in t_hist])
        
        # Also prepare derivative history for neutral DDEs
        if is_neutral:
            y_prime_hist = np.array([hist_prime_func(t) for t in t_hist])
            
            # If y_prime_hist contains scalars, reshape to column vector
            if y_prime_hist.ndim == 1:
                y_prime_hist = y_prime_hist.reshape(-1, 1)
    except Exception as e:
        # Handle any errors in evaluating history functions
        raise ValueError(f"Error evaluating history function: {e}")
    
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
        
    # For neutral DDEs, we also need to get the derivative value at delayed time
    if is_neutral:
        # Start with derivative history
        all_y_prime = y_prime_hist
        
        def get_delayed_derivative(t):
            """Get derivative value at time t from interpolation"""
            # If t is in the past, use history derivative function
            if t <= 0:
                return hist_prime_func(t)
                
            # If t is beyond our computed solution, use the latest value
            if t > all_t[-1]:
                warnings.warn(f"Time {t} is beyond computed solution at {all_t[-1]}")
                return all_y_prime[-1]
                
            # Otherwise use interpolation
            interp = interp1d(all_t, all_y_prime, axis=0, bounds_error=False, fill_value="extrapolate")
            return interp(t)
    
    # Define the ODE function that includes the delay term
    def rhs_with_delay(t, y):
        """RHS function that includes the delay term"""
        # Get delayed state
        y_tau = get_delayed_value(t - τ)
        
        # For neutral DDEs, also get delayed derivative
        if is_neutral:
            y_tau_prime = get_delayed_derivative(t - τ)
        
        # Ensure y is properly shaped
        y_reshaped = y.reshape(-1, 1) if y.ndim == 1 and y.size > 1 else y
        
        # Get derivative from the equations
        # We'll access the first equation's function (assuming one equation)
        if callable(eqns['u']):
            # If it's a function, call it directly
            if is_neutral:
                # For neutral DDEs, pass the delayed derivative too
                result = eqns['u'](t, y_reshaped, y_tau, y_tau_prime, **params)
            else:
                result = eqns['u'](t, y_reshaped, y_tau, **params)
                
            # Ensure result is a numpy array with proper shape
            if not isinstance(result, np.ndarray):
                result = np.array([result]).flatten()
            elif result.ndim > 1:
                result = result.flatten()
                
            return result
        else:
            # Default simple implementation for common cases
            # This handles cases like Mackey-Glass, etc.
            if y.size == 1:  # Single-variable case
                u = y.item() if isinstance(y, np.ndarray) else y
                u_tau = y_tau.item() if isinstance(y_tau, np.ndarray) else y_tau
                
                # Simple implementation for Mackey-Glass as default
                beta = params.get('β', params.get('beta', 0.2))
                gamma = params.get('γ', params.get('gamma', 0.1))
                n = params.get('n', 10)
                
                dydt = beta * u_tau/(1 + u_tau**n) - gamma * u
                return np.array([dydt])  # Return as a numpy array
            else:
                # Multi-variable case (e.g., reaction-diffusion)
                # Just return zeros as a placeholder - this should be overridden by callable eqns
                return np.zeros_like(y)
    
    # Integrate in chunks to better handle the delay term
    current_t = t0
    while current_t < tf:
        # Determine end of this chunk
        chunk_end = min(current_t + chunk_size, tf)
        
        try:
            # Integrate over this chunk
            sol = solve_ivp(
                rhs_with_delay,
                (current_t, chunk_end),
                y0,  # y0 is already flattened above
                method='BDF',  # Backward differentiation formula for stiff problems
                rtol=1e-6,
                atol=1e-8,
                dense_output=True
            )
            
            # Store solution points
            t_points.append(sol.t)
            y_points.append(sol.y.T if sol.y.ndim > 1 else sol.y.reshape(-1, 1))
            
            # Update the interpolation data for state
            all_t = np.append(all_t, sol.t)
            if sol.y.ndim == 1:
                new_y = sol.y.reshape(-1, 1)
            else:
                new_y = sol.y.T
            all_y = np.vstack([all_y, new_y])
            
            # For neutral DDEs, we also need to update derivative history
            if is_neutral:
                # Estimate derivatives from the solution using finite differences
                if len(sol.t) > 1:
                    dt = np.diff(sol.t)
                    dy = np.diff(sol.y, axis=1)
                    derivatives = dy / dt
                    
                    # Add estimated derivatives at new time points (except last point)
                    new_t_prime = sol.t[:-1]
                    if derivatives.ndim == 1:
                        new_y_prime = derivatives.reshape(-1, 1)
                    else:
                        new_y_prime = derivatives.T
                    
                    # For the last point, use forward extrapolation
                    last_deriv = new_y_prime[-1] if new_y_prime.shape[0] > 0 else np.zeros((1, new_y.shape[1]))
                    last_t = sol.t[-1]
                    
                    # Append to all_y_prime
                    all_t_prime = np.append(all_t[:-1], new_t_prime)
                    all_t_prime = np.append(all_t_prime, last_t)
                    all_y_prime = np.vstack([all_y_prime, new_y_prime, last_deriv])
            
            # Update current time and initial condition for next chunk
            current_t = chunk_end
            y0 = sol.y[:, -1] if sol.y.ndim > 1 else np.array([sol.y[-1]])  # Last solution point
            
        except Exception as e:
            # If integration fails, provide helpful error message
            raise ValueError(f"Integration failed at t={current_t}: {e}")
    
    # Combine all solution points
    t_combined = np.concatenate(t_points) if len(t_points) > 0 else np.array([t0])
    
    if len(y_points) > 0:
        # Stack y_points, ensuring consistent shape
        y_combined = np.vstack(y_points)
        
        # Return solution at requested evaluation points using interpolation
        sol_interp = interp1d(t_combined, y_combined, axis=0, bounds_error=False, fill_value="extrapolate")
        y_eval = sol_interp(t_eval)
    else:
        # If no solutions were computed (e.g. immediate error), return empty result
        y_eval = np.zeros((len(t_eval), y0.size))
    
    return t_eval, y_eval
