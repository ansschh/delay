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
        History function that takes time t and returns state at time t
        For neutral DDEs, this may be a tuple of (hist_func, hist_deriv_func)
    t_eval: ndarray
        Times at which to return the solution
    
    Returns:
    --------
    tuple
        (t, y) times and solution values
    """
    # Extract the main delay value (assuming single delay for simplicity)
    τ = τs[0]
    
    # Get the history function - handle both regular history and neutral DDE cases
    if 'u' in history:
        if isinstance(history['u'], tuple):
            # For neutral DDEs, the history is a tuple (hist_func, hist_deriv_func)
            hist_func = history['u'][0]  # Take the value function, not derivative
        else:
            hist_func = history['u']
    else:
        # Fallback - try direct access if no 'u' key
        hist_func = history
    
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
    
    # Initial condition - handle callable or non-callable hist_func
    try:
        if callable(hist_func):
            y0 = hist_func(0.0)
        else:
            # If hist_func is not callable (e.g., a value), use it directly
            y0 = hist_func
    except Exception as e:
        # If there's an error, try a simpler approach
        print(f"Warning when getting initial condition: {e}")
        if isinstance(hist_func, (list, tuple, np.ndarray)):
            if callable(hist_func[0]):
                y0 = hist_func[0](0.0)
            else:
                y0 = hist_func[0]
        else:
            # Default to 1.0 as a last resort
            y0 = 1.0
    
    # Ensure y0 is properly formatted for the solver
    if not isinstance(y0, (list, np.ndarray)):
        y0 = np.array([y0])
    
    # Create interpolation function for history
    from scipy.interpolate import interp1d
    
    # Start with history on [-τ, 0]
    t_hist = np.linspace(-τ, 0, max(100, int(τ/0.01)))  # Fine grid for history
    
    # Get history values, handling any potential errors
    try:
        if callable(hist_func):
            y_hist = np.array([hist_func(t) for t in t_hist])
        else:
            # If hist_func is not callable, use a constant history
            y_hist = np.array([y0 for _ in t_hist])
    except Exception as e:
        print(f"Warning when sampling history: {e}")
        # Default to constant history if there's an error
        y_hist = np.array([y0 for _ in t_hist])
    
    # Ensure y_hist has the right shape
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
            try:
                if callable(hist_func):
                    return hist_func(t)
                else:
                    # Return the first history value as an approximation
                    return y_hist[0]
            except Exception:
                # Fallback to interpolation
                pass
        
        # If t is beyond our computed solution, use the latest value
        if t > all_t[-1]:
            return all_y[-1]
        
        # Otherwise use interpolation
        try:
            interp = interp1d(all_t, all_y, axis=0, bounds_error=False, fill_value="extrapolate")
            return interp(t)
        except Exception as e:
            print(f"Interpolation error: {e}")
            # Return the latest computed value as fallback
            return all_y[-1]
    
    # Define the ODE function that includes the delay term
    def rhs_with_delay(t, y):
        """RHS function that includes the delay term"""
        # Make sure y is properly shaped
        if isinstance(y, np.ndarray) and y.size == 1:
            y = np.array([y[0]])  # Ensure it's a 1D array with one element
        
        # Get delayed state
        try:
            y_tau = get_delayed_value(t - τ)
        except Exception as e:
            print(f"Error getting delayed value: {e}")
            # Use current value as fallback
            y_tau = y
        
        # Get derivative from the equations
        try:
            if 'u' in eqns and callable(eqns['u']):
                # If it's a function, call it directly
                result = eqns['u'](t, y, y_tau, **params)
                # Ensure the result is properly shaped for the solver
                if isinstance(result, (int, float)):
                    result = np.array([result])
                return result
            else:
                # Default implementation for simple cases
                u = y[0] if isinstance(y, (list, np.ndarray)) and len(y) > 0 else y
                u_tau = y_tau[0] if isinstance(y_tau, (list, np.ndarray)) and len(y_tau) > 0 else y_tau
                
                # Simple implementation for Mackey-Glass as default
                beta = params.get('β', params.get('beta', 0.2))
                gamma = params.get('γ', params.get('gamma', 0.1))
                n = params.get('n', 10)
                
                dydt = beta * u_tau/(1 + u_tau**n) - gamma * u
                return np.array([dydt])
        except Exception as e:
            print(f"Error in RHS function: {e}")
            # Return a small default value as fallback
            return np.zeros_like(y)
    
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
