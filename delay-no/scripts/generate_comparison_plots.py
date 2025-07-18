#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def set_plot_style():
    """Set consistent plot style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("colorblind")

def plot_metric_by_model_dataset(df, metric, output_dir, title=None, ylabel=None, ylim=None):
    """
    Plot specified metric for all models and datasets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Results dataframe
    metric : str
        Column name of the metric to plot
    output_dir : str
        Directory to save plot
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    ylim : tuple, optional
        Y-axis limits (min, max)
    """
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(
        data=df,
        x='dataset',
        y=metric,
        hue='model',
        errorbar=None
    )
    
    # Add value annotations on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
        
    # Customize plot
    plt.title(title or f"{metric.replace('_', ' ').title()} by Model and Dataset")
    plt.xlabel("Dataset")
    plt.ylabel(ylabel or metric.replace('_', ' ').title())
    plt.xticks(rotation=45)
    
    if ylim:
        plt.ylim(ylim)
    
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=150)
    plt.close()

def plot_performance_radar(df, metrics, output_dir):
    """
    Create radar plots for model performance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Results dataframe
    metrics : list
        List of metrics to include in radar plot
    output_dir : str
        Directory to save plot
    """
    # Normalize metrics for radar plot
    df_radar = df.copy()
    
    for metric in metrics:
        if metric in ['avg_l2_error', 'avg_spectral_error']:
            # For error metrics, lower is better, so invert
            max_val = df_radar[metric].max()
            df_radar[f"{metric}_norm"] = 1 - (df_radar[metric] / max_val)
        elif metric == 'stability_metric':
            # For stability metric, closer to 1 is better
            df_radar[f"{metric}_norm"] = 1 - np.abs(df_radar[metric] - 1) / max(np.abs(df_radar[metric] - 1).max(), 1)
        elif metric == 'throughput':
            # For throughput, higher is better
            max_val = df_radar[metric].max()
            df_radar[f"{metric}_norm"] = df_radar[metric] / max_val
        else:
            # Default normalization
            min_val = df_radar[metric].min()
            max_val = df_radar[metric].max()
            df_radar[f"{metric}_norm"] = (df_radar[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    
    # Create a radar plot for each dataset
    for dataset in df['dataset'].unique():
        df_dataset = df_radar[df_radar['dataset'] == dataset]
        
        # Set up the radar chart
        num_metrics = len(metrics)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        metric_names = [m.replace('_', ' ').replace('avg ', '').title() for m in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each model
        for model in df_dataset['model'].unique():
            values = df_dataset[df_dataset['model'] == model][[f"{m}_norm" for m in metrics]].values.flatten().tolist()
            values += values[:1]  # Close the polygon
            
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        # Customize chart
        ax.set_theta_offset(np.pi / 2)  # Start at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        
        # Draw y-axis lines
        ax.set_rlabel_position(0)
        ax.set_rticks([0.25, 0.5, 0.75])
        ax.set_rmax(1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f"Model Performance on {dataset.replace('_', ' ').title()}", size=15, y=1.05)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"{dataset}_radar.png"), dpi=150, bbox_inches='tight')
        plt.close()

def create_summary_table(df, output_dir):
    """
    Create and save summary tables
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Results dataframe
    output_dir : str
        Directory to save tables
    """
    # Create pivot tables
    metrics = ['avg_l2_error', 'avg_spectral_error', 'stability_metric', 'throughput']
    
    for metric in metrics:
        pivot = df.pivot_table(
            index='model', 
            columns='dataset', 
            values=metric,
            aggfunc='mean'
        )
        
        # Add average column
        pivot['average'] = pivot.mean(axis=1)
        
        # Save to CSV
        pivot.to_csv(os.path.join(output_dir, f"{metric}_summary.csv"))
    
    # Create summary LaTeX table
    best_model_per_dataset = {}
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        best_model = dataset_df.loc[dataset_df['avg_l2_error'].idxmin()]['model']
        best_model_per_dataset[dataset] = best_model
    
    # Save best model summary
    with open(os.path.join(output_dir, "best_models.txt"), 'w') as f:
        f.write("Best model per dataset (based on L2 error):\n")
        for dataset, model in best_model_per_dataset.items():
            f.write(f"{dataset}: {model}\n")

def plot_heatmaps(df, metrics, output_dir):
    """
    Create heatmaps for model-dataset performance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Results dataframe
    metrics : list
        List of metrics to plot
    output_dir : str
        Directory to save heatmaps
    """
    for metric in metrics:
        plt.figure(figsize=(10, 8))
        
        # Create pivot table
        pivot = df.pivot_table(
            index='model', 
            columns='dataset', 
            values=metric,
            aggfunc='mean'
        )
        
        # Create heatmap
        if metric in ['avg_l2_error', 'avg_spectral_error']:
            # For error metrics, lower is better
            cmap = 'RdYlGn_r'
        else:
            # For other metrics, higher is better
            cmap = 'RdYlGn'
            
        ax = sns.heatmap(
            pivot, 
            annot=True, 
            fmt=".3f",
            cmap=cmap, 
            linewidths=.5
        )
        
        plt.title(f"{metric.replace('_', ' ').title()} Across Models and Datasets")
        plt.tight_layout()
        
        # Save heatmap
        plt.savefig(os.path.join(output_dir, f"{metric}_heatmap.png"), dpi=150)
        plt.close()

def plot_correlation_matrix(df, output_dir):
    """
    Plot correlation matrix between metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Results dataframe
    output_dir : str
        Directory to save plot
    """
    # Select numeric columns
    numeric_cols = ['avg_l2_error', 'avg_spectral_error', 'stability_metric', 'throughput', 'model_parameters']
    
    # Calculate correlation matrix
    corr = df[numeric_cols].corr()
    
    # Plot
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        mask=mask,
        linewidths=.5
    )
    
    plt.title("Correlation Between Metrics")
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "metrics_correlation.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots for Delay-NO benchmarks')
    parser.add_argument('--results_file', type=str, required=True, help='Path to results CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    df = pd.read_csv(args.results_file)
    
    # Set plot style
    set_plot_style()
    
    # Metrics to analyze
    metrics = ['avg_l2_error', 'avg_spectral_error', 'stability_metric', 'throughput']
    
    # Generate individual metric plots
    for metric in metrics:
        plot_metric_by_model_dataset(df, metric, args.output_dir)
    
    # Generate radar plots
    plot_performance_radar(df, metrics, args.output_dir)
    
    # Generate heatmaps
    plot_heatmaps(df, metrics, args.output_dir)
    
    # Generate correlation matrix
    plot_correlation_matrix(df, args.output_dir)
    
    # Create summary tables
    create_summary_table(df, args.output_dir)
    
    print(f"All comparison plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
