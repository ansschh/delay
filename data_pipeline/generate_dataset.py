# generate_dataset.py

import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Import solvers
from solvers.method_of_steps import solve_dde
from solvers.radar5_wrapper import solve_stiff_dde

# Import DDE families
from families import mackey_glass, delayed_logistic, neutral_dde, reaction_diffusion

# Import utility functions
from utils.io import save_dataset, plot_solution, create_train_test_split
from utils.history_generators import cubic_spline_history, fourier_history, filtered_brownian_history

def make_family_dataset(family, N=10000, τ_range=(0.5, 5.0), T=50.0, dt=0.1, 
                        history_type="cubic_spline", output_dir="data"):
    """
    Generate a dataset for a DDE family
    
    Parameters:
    -----------
    family : module
        DDE family module
    N : int, optional
        Number of samples (default=10000)
    τ_range : tuple, optional
        Range of delay values (default=(0.5, 5.0))
    T : float, optional
        Maximum integration time (default=50.0)
    dt : float, optional
        Step size for window (default=0.1)
    history_type : str, optional
        Type of history function to use (default="cubic_spline")
        Options: "cubic_spline", "fourier", "filtered_brownian"
    output_dir : str, optional
        Directory to save output data (default="data")
        
    Returns:
    --------
    list
        List of (history, τ, t, y) tuples
    """
    data = []
    
    print(f"Generating {N} samples for {family.__name__}...")
    
    # Use tqdm for progress tracking
    for i in tqdm(range(N), desc=f"Generating {family.__name__} datasets", ncols=100):
        
        # Sample delay value
        τ = np.random.uniform(*τ_range)
        
        # Generate history function based on type
        if hasattr(family, 'random_history'):
            # Use family-specific history generator if available
            hist = family.random_history(τ)
        else:
            # Use generic history generator
            if history_type == "cubic_spline":
                hist = cubic_spline_history(τ, d=1)
            elif history_type == "fourier":
                hist = fourier_history(τ, d=1)
            elif history_type == "filtered_brownian":
                hist = filtered_brownian_history(τ, d=1)
            else:
                raise ValueError(f"Unknown history type: {history_type}")
        
        # Choose solver based on stiffness flag
        if hasattr(family, 'stiff') and family.stiff:
            t_eval = np.arange(0, T, dt)
            # For neutral DDEs, we need special handling
            if family.__name__ == 'neutral_dde' and isinstance(hist, tuple):
                history_dict = {'u': hist[0]}  # First element is the history function
                t, y = solve_stiff_dde(family.eqns, family.params, [τ], history_dict, t_eval)
            else:
                history_dict = {'u': hist}
                t, y = solve_stiff_dde(family.eqns, family.params, [τ], history_dict, t_eval)
        else:
            # Use method of steps solver
            t, y = solve_dde(family.f, hist, τ, (0, T), dt)
        
        data.append((hist, τ, t, y))
    
    print(f"Done! Generated {len(data)} samples.")
    return data

def generate_all_datasets(config):
    """
    Generate datasets for all specified DDE families
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with generation parameters
    """
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Map family names to modules
    family_map = {
        'mackey_glass': mackey_glass,
        'delayed_logistic': delayed_logistic,
        'neutral_dde': neutral_dde,
        'reaction_diffusion': reaction_diffusion
    }
    
    for family_name in config['families']:
        if family_name not in family_map:
            print(f"Warning: Unknown family '{family_name}'. Skipping.")
            continue
        
        family = family_map[family_name]
        
        try:
            # Generate dataset
            data = make_family_dataset(
                family,
                N=config['N'],
                τ_range=config['τ_range'],
                T=config['T'],
                dt=config['dt'],
                history_type=config['history_type'],
                output_dir=config['output_dir']
            )
            
            # Split into train/test if specified
            if config['τ_split'] is not None:
                train_data, test_data = create_train_test_split(data, τ_split=config['τ_split'])
                
                # Save train/test datasets
                train_filename = os.path.join(config['output_dir'], f"{family_name}_train.pkl")
                test_filename = os.path.join(config['output_dir'], f"{family_name}_test.pkl")
                
                save_dataset(train_data, train_filename)
                save_dataset(test_data, test_filename)
                
                # Optionally plot some examples
                if config['plot_examples']:
                    for dataset, name in [(train_data, 'train'), (test_data, 'test')]:
                        if len(dataset) > 0:
                            idx = np.random.randint(len(dataset))
                            hist_data, τ, t, y = dataset[idx]
                            plot_path = os.path.join(config['output_dir'], f"{family_name}_{name}_example.png")
                            # Use t and y for plotting (skip history function which is now serialized data)
                            plot_solution(t, y, τ, None, 
                                        title=f"{family_name} - {name} example, τ={τ:.2f}",
                                        save_path=plot_path)
                            print(f"Saved example plot to {plot_path}")
            else:
                # Save full dataset
                filename = os.path.join(config['output_dir'], f"{family_name}.pkl")
                save_dataset(data, filename)
                
                # Optionally plot an example
                if config['plot_examples'] and len(data) > 0:
                    idx = np.random.randint(len(data))
                    hist_data, τ, t, y = data[idx]
                    plot_path = os.path.join(config['output_dir'], f"{family_name}_example.png")
                    # Use t and y for plotting (skip history function which is now serialized data)
                    plot_solution(t, y, τ, None, 
                                title=f"{family_name} example, τ={τ:.2f}",
                                save_path=plot_path)
                    print(f"Saved example plot to {plot_path}")
            
            print(f"✓ Successfully generated dataset for {family_name}")
            
        except Exception as e:
            print(f"✗ Failed to generate dataset for {family_name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate DDE datasets for Delay-NO training')
    parser.add_argument('--families', nargs='+', 
                        choices=['mackey_glass', 'delayed_logistic', 'neutral_dde', 'reaction_diffusion'],
                        default=['mackey_glass'],
                        help='DDE families to generate data for')
    parser.add_argument('--N', type=int, default=1000,
                        help='Number of samples per family')
    parser.add_argument('--tau_min', type=float, default=0.5,
                        help='Minimum delay value')
    parser.add_argument('--tau_max', type=float, default=5.0,
                        help='Maximum delay value')
    parser.add_argument('--T', type=float, default=50.0,
                        help='Maximum integration time')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Step size for method of steps')
    parser.add_argument('--history_type', type=str, 
                        choices=['cubic_spline', 'fourier', 'filtered_brownian'],
                        default='cubic_spline',
                        help='Type of history function to use')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save output data')
    parser.add_argument('--tau_split_min', type=float, default=None,
                        help='If specified with tau_split_max, delay values in this range are held out for testing')
    parser.add_argument('--tau_split_max', type=float, default=None,
                        help='If specified with tau_split_min, delay values in this range are held out for testing')
    parser.add_argument('--plot_examples', action='store_true',
                        help='Whether to plot example solutions')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'families': args.families,
        'N': args.N,
        'τ_range': (args.tau_min, args.tau_max),
        'T': args.T,
        'dt': args.dt,
        'history_type': args.history_type,
        'output_dir': args.output_dir,
        'τ_split': (args.tau_split_min, args.tau_split_max) if args.tau_split_min is not None else None,
        'plot_examples': args.plot_examples
    }
    
    generate_all_datasets(config)
