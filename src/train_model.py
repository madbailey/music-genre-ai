import math
import numpy as np
import librosa
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.data_loader import original_load_dataset
from src.augment import augment_audio, time_warp
from src.model import ModelTrainer
from src.bayesianoptimizer import BayesianOptimizationManager, BayesianTrialMonitor
from src.model_progress import ModelHistoryTracker

# Create tracker instance
tracker = ModelHistoryTracker()

class AudioDataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.class_names = ['blues', 'classical', 'country', 'disco', 
                           'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    def mixup_data(self, X_mfccs, X_spectral, X_chroma, y, alpha=0.2):
        """Performs mixup on the input data and their labels."""
        if alpha <= 0:
            return X_mfccs, X_spectral, X_chroma, y
            
        # Validate inputs
        for X, name in [(X_mfccs, 'MFCC'), (X_spectral, 'Spectral'), (X_chroma, 'Chroma')]:
            if len(X.shape) != 3:
                raise ValueError(f"Expected 3D input tensor for {name}, got shape {X.shape}")
        if len(y.shape) != 2:
            raise ValueError(f"Expected 2D one-hot encoded labels, got shape {y.shape}")
        
        # Generate and reshape mixup weights
        weights = np.random.beta(alpha, alpha, len(X_mfccs))
        indices = np.random.permutation(len(X_mfccs))
        X_weights = weights.reshape(len(X_mfccs), 1, 1)
        y_weights = weights.reshape(len(X_mfccs), 1)
        
        # Perform mixup
        X_mfccs_mixed = X_weights * X_mfccs + (1 - X_weights) * X_mfccs[indices]
        X_spectral_mixed = X_weights * X_spectral + (1 - X_weights) * X_spectral[indices]
        X_chroma_mixed = X_weights * X_chroma + (1 - X_weights) * X_chroma[indices]
        y_mixed = y_weights * y + (1 - y_weights) * y[indices]
        
        return X_mfccs_mixed, X_spectral_mixed, X_chroma_mixed, y_mixed

    def process_features(self, mfccs, spectral, chroma, idx):
        """Process individual audio features."""
        try:
            # Process MFCC features
            mfccs = augment_audio(mfccs)
            mfccs = time_warp(mfccs)
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            features = np.vstack([mfccs, delta, delta2])
            
            # Validate shapes
            if features.shape != (39, 130):
                print(f"Warning: Inconsistent MFCC shape {features.shape} for sample {idx}")
                return None
            if spectral.shape != (1, 130):
                print(f"Warning: Inconsistent spectral shape {spectral.shape} for sample {idx}")
                return None
            if chroma.shape != (12, 130):
                print(f"Warning: Inconsistent chroma shape {chroma.shape} for sample {idx}")
                return None
                
            return features, spectral, chroma
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return None

    def load_and_process_dataset(self):
        """Load and process the complete dataset."""
        X_mfccs, X_spectral, X_chroma, y = original_load_dataset()
        
        processed_data = []
        for i in range(len(X_mfccs)):
            result = self.process_features(X_mfccs[i], X_spectral[i], X_chroma[i], i)
            if result is not None:
                processed_data.append(result)
        
        # Unzip processed data
        X_mfccs_processed, X_spectral_processed, X_chroma_processed = zip(*processed_data)
        
        # Convert to numpy arrays
        X_mfccs_processed = np.array(X_mfccs_processed)
        X_spectral_processed = np.array(X_spectral_processed)
        X_chroma_processed = np.array(X_chroma_processed)
        
        # Trim labels to match processed data
        y = y[:len(X_mfccs_processed)]
        
        return X_mfccs_processed, X_spectral_processed, X_chroma_processed, y

    def prepare_data(self):
        """Full data preparation pipeline."""
        # Load and process data
        X_mfccs, X_spectral, X_chroma, y = self.load_and_process_dataset()
        
        # Split data
        splits = train_test_split(
            X_mfccs, X_spectral, X_chroma, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        (X_mfccs_train, X_mfccs_val,
         X_spectral_train, X_spectral_val,
         X_chroma_train, X_chroma_val,
         y_train, y_val) = splits

        # One-hot encode labels
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        # Apply mixup to training data
        X_mfccs_train, X_spectral_train, X_chroma_train, y_train = self.mixup_data(
            X_mfccs_train, X_spectral_train, X_chroma_train, y_train
        )

        # Reshape and add channel dimension
        X_mfccs_train, X_mfccs_val = self._reshape_features(X_mfccs_train, X_mfccs_val)
        X_spectral_train, X_spectral_val = self._reshape_features(X_spectral_train, X_spectral_val)
        X_chroma_train, X_chroma_val = self._reshape_features(X_chroma_train, X_chroma_val)

        # Normalize features
        X_mfccs_train, X_mfccs_val = self._normalize_features(X_mfccs_train, X_mfccs_val)
        X_spectral_train, X_spectral_val = self._normalize_features(X_spectral_train, X_spectral_val)
        X_chroma_train, X_chroma_val = self._normalize_features(X_chroma_train, X_chroma_val)

        return {
            'train': {
                'mfccs': X_mfccs_train,
                'spectral': X_spectral_train,
                'chroma': X_chroma_train,
                'labels': y_train
            },
            'val': {
                'mfccs': X_mfccs_val,
                'spectral': X_spectral_val,
                'chroma': X_chroma_val,
                'labels': y_val
            }
        }

    def _reshape_features(self, train_data, val_data):
        """Reshape features and add channel dimension."""
        train_data = np.transpose(train_data, (0, 2, 1))[..., np.newaxis]
        val_data = np.transpose(val_data, (0, 2, 1))[..., np.newaxis]
        return train_data, val_data

    def _normalize_features(self, train_data, val_data):
        """Normalize features using training data statistics."""
        mean = np.mean(train_data, axis=(0, 1))
        std = np.std(train_data, axis=(0, 1))
        train_norm = (train_data - mean) / (std + 1e-8)
        val_norm = (val_data - mean) / (std + 1e-8)
        return train_norm, val_norm


def main():
    # Initialize data processor
    processor = AudioDataProcessor()
    data = processor.prepare_data()
    
    # Define input shapes and create model trainer
    input_shapes = {
        'mfcc': (130, 39),
        'spectral': (130, 1),
        'chroma': (130, 12)
    }
    trainer = ModelTrainer(input_shapes, num_classes=len(processor.class_names))
    
    # Set up Bayesian optimization
    optimizer = BayesianOptimizationManager(
        trainer.build_model,
        max_trials=10,
        directory='tuning_dir'
    )
    
    # Configure callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Run optimization
    optimizer.search(
        [data['train']['mfccs'], data['train']['spectral'], data['train']['chroma']],
        data['train']['labels'],
        [data['val']['mfccs'], data['val']['spectral'], data['val']['chroma']],
        data['val']['labels'],
        class_names=processor.class_names,
        epochs=75,
        callbacks=callbacks
    )
     # After optimization completes
    best_hp = optimizer.tuner.get_best_hyperparameters()[0]
    model = trainer.build_model(best_hp)
    
    # Training configuration for best model
    model_config = {
        'hyperparameters': best_hp.values,
        'architecture': {
            'input_shapes': input_shapes,
            'num_classes': len(processor.class_names)
        }
    }
    
    # Train best model
    history = model.fit(
        [data['train']['mfccs'], data['train']['spectral'], data['train']['chroma']],
        data['train']['labels'],
        validation_data=(
            [data['val']['mfccs'], data['val']['spectral'], data['val']['chroma']],
            data['val']['labels']
        ),
        epochs=75,
        callbacks=callbacks,
        verbose=1
    )

    # Calculate final metrics
    train_metrics = model.evaluate(
        [data['train']['mfccs'], data['train']['spectral'], data['train']['chroma']],
        data['train']['labels'],
        verbose=0
    )
    val_metrics = model.evaluate(
        [data['val']['mfccs'], data['val']['spectral'], data['val']['chroma']],
        data['val']['labels'],
        verbose=0
    )

    # Package metrics
    metrics_dict = {
        'train': {
            'loss': float(train_metrics[0]),
            'accuracy': float(train_metrics[1]),
            'auc': float(train_metrics[2])
        },
        'val': {
            'loss': float(val_metrics[0]),
            'accuracy': float(val_metrics[1]),
            'auc': float(val_metrics[2])
        }
    }

    # Save training history
    tracker.save_run(history, model_config, 
                    metrics_dict['train'], 
                    metrics_dict['val'])

    # Save best model
    model.save("model/genre_classifier.keras")

    print("\nOptimization complete. Results saved in tuning_results directory.")

if __name__ == "__main__":
    main()