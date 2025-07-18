#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Analysis for Delay Differential Equations (DDEs)
========================================================

This script provides comprehensive analysis and visualization 
of DDE datasets generated for the Delay-NO project.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pickle
from tqdm import tqdm
import argparse

# Configure prettier plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)


def load_dataset(file_path):
    """Load a dataset from pickle file"""
    print(f"Loading dataset: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded dataset with {len(data)} samples")
        return data
    except Exception as e:
        print(f"✗ Error loading dataset: {str(e)}")
        return None


def get_dataset_stats(dataset):
    """Extract basic statistics from dataset"""
    stats = {}
    
    # Sample count
    stats['sample_count'] = len(dataset)
    
    # The dataset structure is (hist, τ, t, y) tuples
    
    # Delay values
    taus = [sample[1] for sample in dataset]  # τ is the second element
    stats['taus'] = np.array(taus)
    
    # Solution dimensions
    solution_dims = []
    for sample in dataset:
        y = sample[3]  # y is the fourth element
        solution_dims.append(y.shape)
    stats['solution_dims'] = solution_dims
    
    # Time points
    stats['t_max'] = np.mean([sample[2][-1] for sample in dataset])  # t is the third element
    stats['t_steps'] = np.mean([len(sample[2]) for sample in dataset])
    
    # Create a dummy params DataFrame with just tau values
    # since we don't have other parameters in the stored data
    params_df = pd.DataFrame({'τ': taus})
    stats['params_df'] = params_df
    
    return stats


def plot_parameter_distributions(stats, family, output_dir):
    """Plot distributions of parameters"""
    params_df = stats['params_df']
    
    # Create figure
    n_params = len(params_df.columns)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each parameter distribution
    for i, param in enumerate(params_df.columns):
        if i < len(axes):
            sns.histplot(params_df[param], kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {param}")
    
    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{family}_parameter_distributions.png"))
    plt.close()


def plot_delay_distribution(stats, family, output_dir):
    """Plot distribution of delay values"""
    taus = stats['taus']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(taus, kde=True)
    plt.title(f"Distribution of Delay Values (τ) - {family}")
    plt.xlabel("τ")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"{family}_delay_distribution.png"))
    plt.close()


def plot_solution_examples(dataset, family, output_dir, n_samples=5):
    """Plot example solutions from the dataset"""
    fig = plt.figure(figsize=(15, 4*n_samples))
    gs = GridSpec(n_samples, 1)
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        ax = fig.add_subplot(gs[i, 0])
        
        # Plot the solution
        # The dataset structure is (hist, τ, t, y) tuples
        hist = sample[0]  # history function or serialized history data
        tau = sample[1]   # delay value
        t = sample[2]     # time points
        y = sample[3]     # solution values
        
        # Handle different dimensions
        if y.ndim == 1:  # 1D solution
            ax.plot(t, y, 'b-', label='u(t)')
            
        elif y.ndim == 2:  # 2D solution (e.g., reaction-diffusion)
            for j in range(min(y.shape[1], 10)):  # Plot up to 10 spatial components
                ax.plot(t, y[:, j], label=f'u{j}(t)' if j < 3 else None)
        
        # Plot a vertical line at t=0 to separate history and forward solution
        ax.axvline(x=0, color='gray', linestyle='--')
        
        # Add delay info
        ax.set_title(f"{family} example {i+1}: τ={tau:.2f}")
        ax.set_xlabel("Time t")
        ax.set_ylabel("u(t)")
        
        # Only show legend for first few components
        if y.ndim == 2 and y.shape[1] > 3:
            ax.legend(loc='upper right', ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{family}_solution_examples.png"))
    plt.close()


def analyze_family_dynamics(dataset, family, output_dir):
    """Analyze dynamical properties of the solutions"""
    # Initialize arrays to store metrics
    max_vals = []
    min_vals = []
    mean_vals = []
    std_vals = []
    oscillation_counts = []
    
    for sample in tqdm(dataset, desc=f"Analyzing {family} dynamics"):
        # The dataset structure is (hist, τ, t, y) tuples
        t = sample[2]  # time points
        y = sample[3]  # solution values
        
        # For multi-dimensional data, analyze the first component
        if y.ndim > 1:
            y_analysis = y[:, 0]
        else:
            y_analysis = y
            
        # Only analyze the forward simulation (t >= 0)
        forward_idx = t >= 0
        y_forward = y_analysis[forward_idx]
        
        # Skip samples with NaN or infinite values
        if np.isnan(y_forward).any() or np.isinf(y_forward).any():
            continue
        
        try:
            # Calculate statistics
            max_vals.append(float(np.max(y_forward)))
            min_vals.append(float(np.min(y_forward)))
            mean_vals.append(float(np.mean(y_forward)))
            std_vals.append(float(np.std(y_forward)))
            
            # Estimate oscillation count (zero crossings of derivative)
            dy = np.diff(y_forward)
            sign_changes = int(np.sum(np.abs(np.diff(np.sign(dy)))))
            oscillation_counts.append(sign_changes)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    # Remove outliers for plotting (values beyond 3 std devs from mean)
    def filter_outliers(data):
        if not data:
            return []
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        return data_array[(data_array > mean - 3*std) & (data_array < mean + 3*std)]
    
    filtered_max_vals = filter_outliers(max_vals)
    filtered_min_vals = filter_outliers(min_vals)
    filtered_std_vals = filter_outliers(std_vals)
    filtered_osc_counts = filter_outliers(oscillation_counts)
    
    # Plot distribution of dynamical properties
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    try:
        if len(filtered_max_vals) > 0:
            sns.histplot(filtered_max_vals, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title(f"{family} - Maximum Values")
        
        if len(filtered_min_vals) > 0:
            sns.histplot(filtered_min_vals, kde=True, ax=axes[0, 1])
            axes[0, 1].set_title(f"{family} - Minimum Values")
        
        if len(filtered_std_vals) > 0:
            sns.histplot(filtered_std_vals, kde=True, ax=axes[1, 0])
            axes[1, 0].set_title(f"{family} - Standard Deviation")
        
        if len(filtered_osc_counts) > 0:
            sns.histplot(filtered_osc_counts, kde=True, ax=axes[1, 1])
            axes[1, 1].set_title(f"{family} - Oscillation Count Estimate")
    except Exception as e:
        print(f"Error creating plots for {family}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{family}_dynamics_analysis.png"))
    plt.close()
    
    # Return computed metrics (original, not filtered)
    return {
        'max_vals': max_vals,
        'min_vals': min_vals, 
        'mean_vals': mean_vals,
        'std_vals': std_vals,
        'oscillation_counts': oscillation_counts
    }


def plot_phase_space(dataset, family, output_dir, n_samples=5):
    """Plot phase space trajectories"""
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(n_samples, n_samples)
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    # We'll use the first component for multi-dimensional systems
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        # The dataset structure is (hist, τ, t, y) tuples
        t = sample[2]  # time points
        y = sample[3]  # solution values
        
        # For multi-dimensional data, analyze the first component
        if y.ndim > 1:
            y_analysis = y[:, 0]
        else:
            y_analysis = y
            
        # Only use the forward simulation (t >= 0)
        forward_idx = t >= 0
        t_forward = t[forward_idx]
        y_forward = y_analysis[forward_idx]
        
        # Create phase portraits with different time lags
        for j, lag in enumerate(range(1, n_samples+1)):
            if i*n_samples + j < n_samples*n_samples:
                ax = fig.add_subplot(gs[i, j])
                
                # Ensure we have enough points for the lag
                if len(y_forward) > lag:
                    ax.plot(y_forward[:-lag], y_forward[lag:], 'b-', alpha=0.7)
                    ax.set_title(f"Lag: {lag}")
                    
                    # If this is the first sample, add more information
                    if i == 0:
                        ax.set_xlabel(f"u(t)")
                        ax.set_ylabel(f"u(t+{lag})")
                    
                    # Add a small marker for the starting point
                    ax.plot(y_forward[0], y_forward[lag], 'ro', markersize=4)
    
    plt.suptitle(f"{family} - Phase Space Analysis with Different Time Lags", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{family}_phase_space.png"))
    plt.close()


def compare_family_dynamics(family_datasets, output_dir):
    """Compare dynamics across families"""
    # Initialize dictionaries to store metrics
    family_metrics = {}
    
    # Calculate metrics for each family
    for family, dataset in family_datasets.items():
        if dataset is None:
            print(f"Skipping {family} family - no dataset loaded")
            continue
            
        print(f"Analyzing dynamics for {family} family...")
        family_metrics[family] = analyze_family_dynamics(dataset, family, output_dir)
    
    if not family_metrics:
        print("No family metrics available for comparison")
        return
    
    # Helper function to filter outliers for better visualization
    def filter_outliers(data_list):
        if not data_list or len(data_list) == 0:
            return data_list
        
        # Convert to numpy array, handling empty lists
        filtered_data = []
        for data in data_list:
            if len(data) == 0:
                filtered_data.append([])
                continue
                
            data_array = np.array(data, dtype=float)
            # Skip arrays with NaN or inf
            if np.isnan(data_array).any() or np.isinf(data_array).any():
                filtered_data.append([])
                continue
                
            # Apply percentile-based filtering (more robust than std-dev for extreme outliers)
            q1, q3 = np.percentile(data_array, [1, 99])  
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_data.append(data_array[(data_array >= lower_bound) & (data_array <= upper_bound)])
        
        return filtered_data
    
    try:
        # Create comparison plots
        plt.figure(figsize=(12, 8))
        
        # Prepare filtered data for plotting
        std_data = [family_metrics[family]['std_vals'] for family in family_metrics.keys()]
        osc_data = [family_metrics[family]['oscillation_counts'] for family in family_metrics.keys()]
        max_data = [family_metrics[family]['max_vals'] for family in family_metrics.keys()]
        min_data = [family_metrics[family]['min_vals'] for family in family_metrics.keys()]
        
        # Filter outliers
        std_data_filtered = filter_outliers(std_data)
        osc_data_filtered = filter_outliers(osc_data)
        max_data_filtered = filter_outliers(max_data)
        min_data_filtered = filter_outliers(min_data)
        
        # Only include families with valid data
        valid_families = []
        valid_std_data = []
        valid_osc_data = []
        valid_max_data = []
        valid_min_data = []
        
        for i, family in enumerate(family_metrics.keys()):
            if (len(std_data_filtered[i]) > 0 and 
                len(osc_data_filtered[i]) > 0 and
                len(max_data_filtered[i]) > 0 and
                len(min_data_filtered[i]) > 0):
                valid_families.append(family)
                valid_std_data.append(std_data_filtered[i])
                valid_osc_data.append(osc_data_filtered[i])
                valid_max_data.append(max_data_filtered[i])
                valid_min_data.append(min_data_filtered[i])
        
        if not valid_families:
            print("No valid family data for comparison plots")
            return
            
        # Box plot of standard deviations
        plt.subplot(2, 2, 1)
        plt.boxplot(valid_std_data, labels=valid_families)
        plt.title('Standard Deviation Comparison')
        plt.xticks(rotation=45)
        
        # Box plot of oscillation counts
        plt.subplot(2, 2, 2)
        plt.boxplot(valid_osc_data, labels=valid_families)
        plt.title('Oscillation Count Comparison')
        plt.xticks(rotation=45)
        
        # Box plot of maximum values
        plt.subplot(2, 2, 3)
        plt.boxplot(valid_max_data, labels=valid_families)
        plt.title('Maximum Value Comparison')
        plt.xticks(rotation=45)
        
        # Box plot of minimum values
        plt.subplot(2, 2, 4)
        plt.boxplot(valid_min_data, labels=valid_families)
        plt.title('Minimum Value Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "family_dynamics_comparison.png"))
        plt.close()
        
        print(f"✓ Successfully created family comparison plots")
        
    except Exception as e:
        print(f"Error creating comparison plots: {e}")
        import traceback
        traceback.print_exc()


def analyze_train_test_split(family, data_dir):
    """Analyze train-test split for a family in the combined dataset"""
    # Path to train and test datasets
    train_path = os.path.join(data_dir, 'combined', f"{family}_train.pkl")
    test_path = os.path.join(data_dir, 'combined', f"{family}_test.pkl")
    
    train_data = load_dataset(train_path)
    test_data = load_dataset(test_path)
    
    if train_data is None or test_data is None:
        return
    
    # Extract delay values - the dataset structure is (hist, τ, t, y) tuples
    train_taus = [sample[1] for sample in train_data]  # τ is the second element
    test_taus = [sample[1] for sample in test_data]    # τ is the second element
    
    # Plot delay distribution for train and test
    plt.figure(figsize=(10, 6))
    sns.histplot(train_taus, kde=True, color='blue', label='Train')
    sns.histplot(test_taus, kde=True, color='red', label='Test')
    plt.title(f"{family} - Delay Distribution in Train/Test Split")
    plt.xlabel("τ")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(data_dir, f"{family}_train_test_split.png"))
    plt.close()
    
    # Return statistics
    return {
        'train_count': len(train_data),
        'test_count': len(test_data),
        'train_taus': train_taus,
        'test_taus': test_taus,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze DDE datasets")
    parser.add_argument('--data_dir', default='data', help='Directory containing the datasets')
    parser.add_argument('--output_dir', default='analysis', help='Directory to save analysis results')
    parser.add_argument('--families', nargs='+', 
                        default=['mackey_glass', 'delayed_logistic', 'neutral_dde', 'reaction_diffusion'],
                        help='DDE families to analyze')
    parser.add_argument('--samples', type=int, default=5, help='Number of example samples to visualize')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each family
    processed_families = []
    family_datasets = {}
    
    for family in args.families:
        print(f"\n{'='*50}\nAnalyzing {family} family\n{'='*50}")
        
        # Path to dataset
        dataset_path = os.path.join(args.data_dir, family, f"{family}.pkl")
        
        # Check if dataset file exists
        if not os.path.exists(dataset_path):
            print(f"\nWarning: Dataset file {dataset_path} not found. Skipping {family}.")
            continue
        
        # Load dataset
        dataset = load_dataset(dataset_path)
        if dataset is None:
            continue
            
        # Store dataset for later comparison
        family_datasets[family] = dataset
        processed_families.append(family)
        
        try:
            # Get basic statistics
            stats = get_dataset_stats(dataset)
            print(f"\nBasic statistics for {family}:")
            print(f"- Sample count: {stats['sample_count']}")
            print(f"- Mean time span: [0, {stats['t_max']:.2f}]")
            print(f"- Mean time steps: {stats['t_steps']:.1f}")
            print(f"- Solution dimensions: {stats['solution_dims'][0]}")
            
            # Create visualizations
            print("\nGenerating visualizations...")
            
            # Parameter distributions
            plot_parameter_distributions(stats, family, args.output_dir)
            
            # Delay distribution
            plot_delay_distribution(stats, family, args.output_dir)
            
            # Solution examples
            plot_solution_examples(dataset, family, args.output_dir, n_samples=args.samples)
            
            # Phase space analysis
            plot_phase_space(dataset, family, args.output_dir, n_samples=3)
            
            # Dynamics analysis
            analyze_family_dynamics(dataset, family, args.output_dir)
            
            # Analyze train-test split
            split_stats = analyze_train_test_split(family, args.data_dir)
            if split_stats:
                train_pct = split_stats['train_count'] / (split_stats['train_count'] + split_stats['test_count']) * 100
                test_pct = 100 - train_pct
                print(f"\nTrain-test split analysis:")
                print(f"- Train samples: {split_stats['train_count']} ({train_pct:.1f}%)")
                print(f"- Test samples: {split_stats['test_count']} ({test_pct:.1f}%)")
        except Exception as e:
            print(f"\nError analyzing {family}: {e}")
            import traceback
            traceback.print_exc()
            
    # Compare families if we have more than one
    if len(processed_families) > 1:
        print("\nGenerating family comparison visualizations...")
        compare_family_dynamics(family_datasets, args.output_dir)
    else:
        print("\nNot enough families processed for comparison.")
    
    print(f"\nAnalysis complete. Results saved in {args.output_dir}/")
    

if __name__ == "__main__":
    main()
