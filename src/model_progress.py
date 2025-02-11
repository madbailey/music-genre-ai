# src/model_progress.py
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import json
import datetime

class ModelHistoryTracker:
    def __init__(self, base_dir='/app/model_history'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.history_file = self.base_dir / 'training_history.json'
        self.runs = self._load_history()

    def _load_history(self):
        """Load training history from JSON file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
        
    def save_run(self, history, model_config, train_metrics, val_metrics):
        """Save a training run with its configuration and metrics."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy values to Python types for JSON serialization
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        
        run_data = {
            'timestamp': timestamp,
            'model_config': model_config,
            'history': history_dict,
            'final_metrics': {
                'train': {k: float(v) for k, v in train_metrics.items()},
                'val': {k: float(v) for k, v in val_metrics.items()}
            }
        }
        
        self.runs.append(run_data)
        
        # Save to JSON file
        with open(self.history_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
            
        # Save plots for this run
        self._save_run_plots(run_data, timestamp)
        
        print(f"\nSaved run data for {timestamp}")

    def _save_run_plots(self, run_data, timestamp):
        """Generate and save plots for a specific run."""
        plots_dir = self.base_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(run_data['history']['accuracy'], label='Training')
        plt.plot(run_data['history']['val_accuracy'], label='Validation')
        plt.title(f'Model Accuracy - {timestamp}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(run_data['history']['loss'], label='Training')
        plt.plot(run_data['history']['val_loss'], label='Validation')
        plt.title(f'Model Loss - {timestamp}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = plots_dir / f'run_{timestamp}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved training plots to {save_path}")

    def plot_accuracy_comparison(self, last_n=5):
        """Plot validation accuracy comparison of the last N runs."""
        if not self.runs:
            print("No runs to compare yet.")
            return
            
        plt.figure(figsize=(12, 6))
        
        for run in self.runs[-last_n:]:
            timestamp = run['timestamp']
            val_acc = run['history']['val_accuracy']
            plt.plot(val_acc, label=f'Run {timestamp}')
        
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.base_dir / 'accuracy_comparison.png')
        plt.close()
        
    def print_summary(self, last_n=5):
        """Print summary of the last N runs."""
        if not self.runs:
            print("No runs to summarize yet.")
            return
            
        print(f"\nSummary of last {min(last_n, len(self.runs))} runs:")
        print("-" * 80)
        print(f"{'Timestamp':^20} | {'Final Val Acc':^12} | {'Final Val AUC':^12} | {'Best Val Acc':^12}")
        print("-" * 80)
        
        for run in self.runs[-last_n:]:
            timestamp = run['timestamp']
            final_val_metrics = run['final_metrics']['val']
            best_val_acc = max(run['history']['val_accuracy'])
            
            print(f"{timestamp:^20} | {final_val_metrics['accuracy']:^12.4f} | "
                  f"{final_val_metrics['auc']:^12.4f} | {best_val_acc:^12.4f}")