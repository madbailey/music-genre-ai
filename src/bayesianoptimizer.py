import keras_tuner as kt
import tensorflow as tf
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import datetime
import os
import traceback  # Add this for better error reporting

# Ensure non-interactive backend for matplotlib
plt.switch_backend('agg')

class BayesianTrialMonitor(tf.keras.callbacks.Callback):
    def __init__(self, trial_num, X_train, y_train, X_val, y_val, class_names):
        super().__init__()
        self.trial_num = trial_num
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        self.output_dir = Path(f'tuning_results/trial_{trial_num}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.best_val_acc = 0
        
    def _convert_to_serializable(self, obj):
        """Convert TensorFlow/NumPy types to Python native types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif tf.is_tensor(obj):
            return float(obj.numpy())
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Convert logs to serializable format
        serializable_logs = self._convert_to_serializable(logs)
        self.history.append(serializable_logs)
        
        # Save if best validation accuracy
        current_val_acc = serializable_logs.get('val_accuracy', 0)
        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self._save_visualizations(epoch, serializable_logs)
    
    def _save_visualizations(self, epoch, logs):
        try:
            # 1. Learning Curves
            plt.figure(figsize=(15, 5))
            
            # Plot accuracies
            plt.subplot(1, 2, 1)
            train_acc = [h['accuracy'] for h in self.history]
            val_acc = [h['val_accuracy'] for h in self.history]
            plt.plot(train_acc, label='Train')
            plt.plot(val_acc, label='Validation')
            plt.title(f'Trial {self.trial_num} Accuracy\nBest Val Acc: {self.best_val_acc:.4f}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Plot losses
            plt.subplot(1, 2, 2)
            train_loss = [h['loss'] for h in self.history]
            val_loss = [h['val_loss'] for h in self.history]
            plt.plot(train_loss, label='Train')
            plt.plot(val_loss, label='Validation')
            plt.title(f'Trial {self.trial_num} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'learning_curves_epoch_{epoch}.png')
            plt.close()
            
            # 2. Confusion Matrices
            y_train_pred = self.model.predict(self.X_train)
            y_val_pred = self.model.predict(self.X_val)
            
            plt.figure(figsize=(20, 8))
            
            # Training confusion matrix
            plt.subplot(1, 2, 1)
            cm_train = confusion_matrix(self.y_train.argmax(axis=1), y_train_pred.argmax(axis=1))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title(f'Trial {self.trial_num} Training Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Validation confusion matrix
            plt.subplot(1, 2, 2)
            cm_val = confusion_matrix(self.y_val.argmax(axis=1), y_val_pred.argmax(axis=1))
            sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title(f'Trial {self.trial_num} Validation Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'confusion_matrices_epoch_{epoch}.png')
            plt.close()
            
            # 3. Per-Class Analysis
            train_report = classification_report(
                self.y_train.argmax(axis=1),
                y_train_pred.argmax(axis=1),
                target_names=self.class_names,
                output_dict=True,
                zero_division=0  # Handle zero division gracefully
            )
            
            val_report = classification_report(
                self.y_val.argmax(axis=1),
                y_val_pred.argmax(axis=1),
                target_names=self.class_names,
                output_dict=True,
                zero_division=0  # Handle zero division gracefully
            )
            
            # Plot per-class F1 scores
            plt.figure(figsize=(12, 6))
            x = np.arange(len(self.class_names))
            width = 0.35
            
            train_f1 = [train_report[genre]['f1-score'] for genre in self.class_names]
            val_f1 = [val_report[genre]['f1-score'] for genre in self.class_names]
            
            plt.bar(x - width/2, train_f1, width, label='Train')
            plt.bar(x + width/2, val_f1, width, label='Validation')
            
            plt.xlabel('Genre')
            plt.ylabel('F1 Score')
            plt.title(f'Trial {self.trial_num} Per-Class F1 Scores')
            plt.xticks(x, self.class_names, rotation=45)
            plt.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'per_class_f1_epoch_{epoch}.png')
            plt.close()
            
            # Save metrics to JSON
            metrics = {
                'epoch': epoch,
                'trial': self.trial_num,
                'train_metrics': self._convert_to_serializable(train_report),
                'val_metrics': self._convert_to_serializable(val_report),
                'history': self.history
            }
            
            with open(self.output_dir / f'metrics_epoch_{epoch}.json', 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            print(f"Error in _save_visualizations: {str(e)}")
            traceback.print_exc()
    

class BayesianOptimizationManager:
    def __init__(self, model_builder, max_trials=10, directory='tuning_dir'):
        self.max_trials = max_trials  # Store max_trials as instance variable
        self.tuner = kt.BayesianOptimization(
            model_builder,
            objective='val_accuracy',
            max_trials=max_trials,
            directory=directory,
            project_name='audio_classifier'
        )
        self.results_dir = Path('tuning_results')
        self.results_dir.mkdir(exist_ok=True)
        self.trial_monitors = []
        
    def search(self, X_train, y_train, X_val, y_val, class_names, **kwargs):
        """Run the Bayesian optimization search"""
        current_trial = 1
        
        while current_trial <= self.max_trials:  # Use the instance variable
            print(f"\nStarting Trial {current_trial}/{self.max_trials}")
            
            # Create trial monitor
            monitor = BayesianTrialMonitor(
                current_trial,
                X_train,
                y_train,
                X_val,
                y_val,
                class_names
            )
            
            # Add monitor to callbacks
            callbacks = kwargs.get('callbacks', [])
            callbacks.append(monitor)
            kwargs['callbacks'] = callbacks
            
            # Run single trial
            self.tuner.search(
                x=X_train,
                y=y_train,
                validation_data=(X_val, y_val),
                **kwargs
            )
            
            self.trial_monitors.append(monitor)
            
            # Save trial summary
            self._save_trial_summary(current_trial)
            current_trial += 1
        
        # After all trials, save and plot final results
        self._save_final_results()
    
    def _save_trial_summary(self, trial_num):
        """Save summary for a single trial"""
        try:
            best_hp = self.tuner.get_best_hyperparameters(1)[0]
            
            summary = {
                'trial': trial_num,
                'best_val_accuracy': float(self.trial_monitors[-1].best_val_acc),
                'hyperparameters': best_hp.values,
            }
            
            with open(self.results_dir / f'trial_{trial_num}_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Error saving trial summary: {str(e)}")
            
    def _save_final_results(self):
        """Save and visualize results from all trials"""
        try:
            # Compile results across all trials
            trial_results = []
            for trial_num in range(1, self.max_trials + 1):
                try:
                    with open(self.results_dir / f'trial_{trial_num}_summary.json', 'r') as f:
                        trial_results.append(json.load(f))
                except FileNotFoundError:
                    print(f"Warning: Results for trial {trial_num} not found")
                    continue
            
            if not trial_results:
                print("No trial results found to analyze")
                return
            
            # Plot trial performances
            plt.figure(figsize=(10, 6))
            accuracies = [trial['best_val_accuracy'] for trial in trial_results]
            plt.plot(range(1, len(accuracies) + 1), accuracies, 'bo-')
            plt.title('Best Validation Accuracy per Trial')
            plt.xlabel('Trial Number')
            plt.ylabel('Best Validation Accuracy')
            plt.grid(True)
            plt.savefig(self.results_dir / 'trial_performances.png')
            plt.close()
            
            # Save overall best results
            best_trial = max(trial_results, key=lambda x: x['best_val_accuracy'])
            with open(self.results_dir / 'best_results.json', 'w') as f:
                json.dump({
                    'best_trial': best_trial,
                    'all_trials': trial_results
                }, f, indent=2)
            
            # Print summary
            print("\nOptimization Results Summary:")
            print(f"Best validation accuracy: {best_trial['best_val_accuracy']:.4f}")
            print(f"Found in trial: {best_trial['trial']}")
            print("\nBest hyperparameters:")
            for param, value in best_trial['hyperparameters'].items():
                print(f"{param}: {value}")
                
        except Exception as e:
            print(f"Error saving final results: {str(e)}")

    