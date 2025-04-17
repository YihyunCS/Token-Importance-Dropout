import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_experiment_data(experiment_name):
    """Load experiment data from CSV file."""
    log_file = os.path.join("logs", f"{experiment_name}.csv")
    if not os.path.exists(log_file):
        print(f"Error: Log file for experiment '{experiment_name}' not found.")
        return None
    
    return pd.read_csv(log_file)

def plot_metrics_comparison(baseline_data, experiment_data, output_dir="plots"):
    """Create comparison plots for baseline vs experiment."""
    if baseline_data is None or experiment_data is None:
        print("Error: Missing data for plotting.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot
    metrics = [
        {'name': 'Training Loss', 'baseline_col': 'train_loss', 'experiment_col': 'train_loss', 'ylabel': 'Loss'},
        {'name': 'Training BPC', 'baseline_col': 'train_bpc', 'experiment_col': 'train_bpc', 'ylabel': 'Bits per Character'},
        {'name': 'Training Perplexity', 'baseline_col': 'train_perplexity', 'experiment_col': 'train_perplexity', 'ylabel': 'Perplexity'},
        {'name': 'Validation Loss', 'baseline_col': 'val_loss', 'experiment_col': 'val_loss', 'ylabel': 'Loss'},
        {'name': 'Validation BPC', 'baseline_col': 'val_bpc', 'experiment_col': 'val_bpc', 'ylabel': 'Bits per Character'},
        {'name': 'Validation Perplexity', 'baseline_col': 'val_perplexity', 'experiment_col': 'val_perplexity', 'ylabel': 'Perplexity'},
    ]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot baseline
        baseline_steps = baseline_data['step']
        baseline_values = baseline_data[metric['baseline_col']]
        # Remove None/NaN values for validation metrics
        mask = pd.notna(baseline_values)
        plt.plot(baseline_steps[mask], baseline_values[mask], label='Baseline', color='blue', marker='o', markersize=4)
        
        # Plot experiment
        experiment_steps = experiment_data['step']
        experiment_values = experiment_data[metric['experiment_col']]
        # Remove None/NaN values for validation metrics
        mask = pd.notna(experiment_values)
        plt.plot(experiment_steps[mask], experiment_values[mask], label='Token Importance Dropout', color='red', marker='x', markersize=4)
        
        plt.xlabel('Training Steps')
        plt.ylabel(metric['ylabel'])
        plt.title(f'{metric["name"]} Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        filename = f"{metric['name'].lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also plot dropout rate for experiment
    if 'dropout_rate' in experiment_data.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(experiment_data['step'], experiment_data['dropout_rate'], color='purple', marker='o', markersize=4)
        plt.xlabel('Training Steps')
        plt.ylabel('Dropout Rate')
        plt.title('Token Importance Dropout Rate')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'dropout_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}/")

def main():
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python plot_results.py <baseline_experiment> <tid_experiment>")
        sys.exit(1)
        
    baseline_name = sys.argv[1]
    experiment_name = sys.argv[2]
    
    # Load data
    print(f"Loading baseline data from {baseline_name}...")
    baseline_data = load_experiment_data(baseline_name)
    
    print(f"Loading experiment data from {experiment_name}...")
    experiment_data = load_experiment_data(experiment_name)
    
    # Plot comparison
    plot_metrics_comparison(baseline_data, experiment_data)

if __name__ == "__main__":
    main() 