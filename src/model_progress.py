from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import json

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
        
    def _save_run_plots(self, run_data, timestamp):
        """Generate and save plots for a specific run."""
        # Create plots directory if it doesn't exist
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

def analyze_model_performance(model, history, X_train, y_train, X_val, y_val, class_names):
    """Comprehensive model analysis with visualizations."""
    # Create output directory
    output_dir = Path('/app/model_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png')
    plt.close()
    
    # 2. Plot confusion matrices
    plt.figure(figsize=(20, 8))
    
    # Training confusion matrix
    plt.subplot(1, 2, 1)
    y_train_pred = model.predict(X_train)
    cm_train = confusion_matrix(y_train.argmax(axis=1), y_train_pred.argmax(axis=1))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Training Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Validation confusion matrix
    plt.subplot(1, 2, 2)
    y_val_pred = model.predict(X_val)
    cm_val = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Validation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png')
    plt.close()
    
    print("\nSaved visualization files to:", output_dir)
    
    # Print classification reports
    print("\nTraining Classification Report:")
    print(classification_report(y_train.argmax(axis=1), 
                              y_train_pred.argmax(axis=1), 
                              target_names=class_names))
    
    print("\nValidation Classification Report:")
    print(classification_report(y_val.argmax(axis=1), 
                              y_val_pred.argmax(axis=1), 
                              target_names=class_names))