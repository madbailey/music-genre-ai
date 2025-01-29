

import keras_tuner as kt

class CustomTuner(kt.Tuner):  # Assuming you're using kt = keras_tuner
    def on_trial_end(self, trial):
        print(f"\nTrial {trial.trial_id} completed:")
        print(f"Val Accuracy: {trial.best_metrics.get('val_accuracy', 'N/A')}")
        print(f"Best val_accuracy so far: {self.best_metrics.get('val_accuracy', 'N/A')}")
        print("-" * 50)
        super().on_trial_end(trial)

    def on_search_end(self, *args, **kwargs):
        print("\nOptimization Results Summary:")
        print(f"Best validation accuracy: {self.best_metrics.get('val_accuracy', 'N/A')}")
        print("Best hyperparameters:")
        for param, value in self.best_hyperparameters.values.items():
            print(f"{param}: {value}")
        print("\nOptimization complete.")
        super().on_search_end(*args, **kwargs)
    def save_tuning_results(self, tuner):
        """Save hyperparameter tuning results."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        tuning_data = {
            'timestamp': timestamp,
            'best_val_accuracy': float(tuner.best_metrics.get('val_accuracy', 0)),
            'best_hyperparameters': tuner.best_hyperparameters.values,
            'trials_summary': [
                {
                    'trial_id': trial.trial_id,
                    'val_accuracy': float(trial.best_metrics.get('val_accuracy', 0)),
                    'hyperparameters': trial.hyperparameters.values
                }
                for trial in tuner.trials
            ]
        }
        
        # Save to separate JSON file
        tuning_file = self.base_dir / 'tuning_history.json'
        if tuning_file.exists():
            with open(tuning_file, 'r') as f:
                tuning_history = json.load(f)
        else:
            tuning_history = []
        
        tuning_history.append(tuning_data)
        
        with open(tuning_file, 'w') as f:
            json.dump(tuning_history, f, indent=2)
        
        # Plot tuning results
        plt.figure(figsize=(10, 6))
        accuracies = [trial['val_accuracy'] for trial in tuning_data['trials_summary']]
        plt.plot(accuracies, marker='o')
        plt.title('Hyperparameter Tuning Progress')
        plt.xlabel('Trial')
        plt.ylabel('Validation Accuracy')
        plt.grid(True)
        plt.savefig(self.base_dir / f'tuning_results_{timestamp}.png')
        plt.close()
        
        print(f"\nTuning results saved. Best validation accuracy: {tuning_data['best_val_accuracy']:.4f}")