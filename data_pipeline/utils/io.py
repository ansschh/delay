# utils/io.py

import os
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

def save_dataset(data, filename, format='pickle'):
    """
    Save a dataset to disk
    
    Parameters:
    -----------
    data : list or dict
        Dataset to save
    filename : str
        Path to save the dataset
    format : str, optional
        Format to save the dataset ('pickle' or 'json') (default='pickle')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Process the data to make it serializable
    processed_data = []
    
    for item in data:
        # Each item is typically (hist, τ, t, y)
        hist_func, tau, t_values, y_values = item
        
        # Sample the history function on a grid to make it serializable
        t_hist = np.linspace(-tau, 0, 100)  # 100 points in history interval
        try:
            # Try to sample the history function
            if isinstance(hist_func, tuple) and len(hist_func) == 2 and callable(hist_func[0]):
                # This is for neutral DDEs with (hist, hist_prime)
                hist_samples = np.array([hist_func[0](ti) for ti in t_hist])
                hist_prime_samples = np.array([hist_func[1](ti) for ti in t_hist])
                hist_data = {
                    'type': 'neutral_history',
                    't': t_hist.tolist(),
                    'y': hist_samples.tolist(),
                    'y_prime': hist_prime_samples.tolist()
                }
            elif callable(hist_func):
                # Regular history function
                hist_samples = np.array([hist_func(ti) for ti in t_hist])
                hist_data = {
                    'type': 'history',
                    't': t_hist.tolist(),
                    'y': hist_samples.tolist()
                }
            else:
                # If it's not callable, just store it as is
                hist_data = {'type': 'unknown', 'data': str(hist_func)}
        except Exception as e:
            # If sampling fails, just store a placeholder
            hist_data = {'type': 'error', 'error': str(e)}
        
        # Create the processed item
        processed_item = (hist_data, tau, t_values, y_values)
        processed_data.append(processed_item)
    
    if format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(processed_data, f)
    elif format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_data = []
        for item in processed_data:
            hist_data, tau, t_values, y_values = item
            json_item = [
                hist_data,
                tau,
                t_values.tolist() if isinstance(t_values, np.ndarray) else t_values,
                y_values.tolist() if isinstance(y_values, np.ndarray) else y_values
            ]
            json_data.append(json_item)
            
        with open(filename, 'w') as f:
            json.dump(json_data, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved {len(processed_data)} samples to {filename}")

def load_dataset(filename):
    """
    Load a dataset from disk
    
    Parameters:
    -----------
    filename : str
        Path to the dataset file
        
    Returns:
    --------
    list or dict
        Loaded dataset
    """
    if filename.endswith('.pkl') or filename.endswith('.pickle'):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

def plot_solution(t, y, τ, history=None, title=None, save_path=None):
    """
    Plot a DDE solution
    
    Parameters:
    -----------
    t : ndarray
        Time points
    y : ndarray
        Solution values
    τ : float
        Delay value
    history : callable or dict, optional
        History function or serialized history data (default=None)
    title : str, optional
        Plot title (default=None)
    save_path : str, optional
        Path to save the figure (default=None)
    """
    plt.figure(figsize=(10, 6))
    
    # Convert arrays to numpy if they're lists
    if isinstance(t, list):
        t = np.array(t)
    if isinstance(y, list):
        y = np.array(y)
    
    # Ensure y has the right shape for plotting
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.flatten()
    
    # Plot solution
    plt.plot(t, y, 'b-', label='Solution u(t)')
    
    # Plot history if provided
    if history is not None:
        if callable(history):
            # If history is a function, sample it
            t_hist = np.linspace(-τ, 0, 100)
            try:
                y_hist = np.vstack([history(ti) for ti in t_hist])
                plt.plot(t_hist, y_hist, 'r--', label='History h(t)')
            except Exception as e:
                print(f"Could not plot history function: {e}")
        elif isinstance(history, dict) and 'type' in history:
            # If history is our serialized format
            if history['type'] == 'history' and 't' in history and 'y' in history:
                t_hist = history['t']
                y_hist = history['y']
                plt.plot(t_hist, y_hist, 'r--', label='History h(t)')
            elif history['type'] == 'neutral_history' and 't' in history and 'y' in history:
                t_hist = history['t']
                y_hist = history['y']
                plt.plot(t_hist, y_hist, 'r--', label='History h(t)')
    
    # Add vertical line at t=0 to separate history and solution
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Add title
    if title:
        plt.title(title)
    else:
        plt.title(f'DDE Solution with τ={τ:.3f}')
    
    plt.xlabel('Time t')
    plt.ylabel('u(t)')
    plt.legend()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure to prevent too many plots being shown
    # This is important for batch processing
    plt.close()

def create_train_test_split(data, test_ratio=0.2, τ_split=None):
    """
    Split dataset into training and test sets
    
    Parameters:
    -----------
    data : list
        Dataset to split
    test_ratio : float, optional
        Ratio of test data (default=0.2)
    τ_split : tuple, optional
        Delay range to hold out for test set, e.g. (2.0, 3.0)
        If provided, all samples with τ in this range go to test set
        
    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    if τ_split is not None:
        # Split based on delay value
        τ_min, τ_max = τ_split
        train_data = [d for d in data if not (τ_min <= d[1] <= τ_max)]
        test_data = [d for d in data if τ_min <= d[1] <= τ_max]
    else:
        # Random split
        indices = np.random.permutation(len(data))
        test_size = int(test_ratio * len(data))
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
    
    return train_data, test_data
