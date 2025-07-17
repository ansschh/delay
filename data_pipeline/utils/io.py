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
    
    if format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(data, list):
            json_data = []
            for item in data:
                if isinstance(item, tuple):
                    json_item = []
                    for elem in item:
                        if callable(elem):  # Handle history functions
                            # For history functions, we can't directly serialize them
                            # We'll just record that it was a function
                            json_item.append({'type': 'function', 'info': 'history_function'})
                        elif isinstance(elem, np.ndarray):
                            json_item.append(elem.tolist())
                        else:
                            json_item.append(elem)
                    json_data.append(json_item)
                else:
                    json_data.append(item)
            
            with open(filename, 'w') as f:
                json.dump(json_data, f)
        else:
            raise ValueError("JSON format only supports list data for now")
    else:
        raise ValueError(f"Unsupported format: {format}")

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
    history : callable, optional
        History function (default=None)
    title : str, optional
        Plot title (default=None)
    save_path : str, optional
        Path to save the figure (default=None)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot solution
    plt.plot(t, y, 'b-', label='Solution u(t)')
    
    # Plot history if provided
    if history is not None:
        t_hist = np.linspace(-τ, 0, 100)
        y_hist = np.vstack([history(ti) for ti in t_hist])
        plt.plot(t_hist, y_hist, 'r--', label='History h(t)')
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
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
    
    plt.show()

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
