import math
import numpy as np
import librosa
from src.data_loader import original_load_dataset
from src.augment import augment_audio
from src.model import create_model
from src.visualize import analyze_model_performance
from src.model_progress import ModelHistoryTracker
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def mixup_data(X, y, alpha=0.2):
    """Performs mixup on the input data and their labels.
    
    Args:
        X: Input data, shape (samples, time_steps, features, channels)
        y: One-hot encoded labels, shape (samples, num_classes)
        alpha: Mixup interpolation strength
    
    Returns:
        Tuple of mixed input data and labels
    """
    if alpha <= 0:
        return X, y
        
    # Validate inputs
    if len(X.shape) != 4:
        raise ValueError(f"Expected 4D input tensor, got shape {X.shape}")
    if len(y.shape) != 2:
        raise ValueError(f"Expected 2D one-hot encoded labels, got shape {y.shape}")
    
    # Generate mixup weights
    weights = np.random.beta(alpha, alpha, len(X))
    indices = np.random.permutation(len(X))
    
    # Reshape weights for broadcasting
    X_weights = weights.reshape(len(X), 1, 1, 1)
    y_weights = weights.reshape(len(X), 1)
    
    # Perform mixup
    X_mixed = X_weights * X + (1 - X_weights) * X[indices]
    y_mixed = y_weights * y + (1 - y_weights) * y[indices]
    
    return X_mixed, y_mixed

def load_dataset():
    """
    Load and augment dataset with consistent shapes
    Returns:
        X: numpy array of shape (samples, 39, 130)
        y: numpy array of shape (samples,)
    """
    X, y = original_load_dataset()
    X_augmented = []
    
    # Process each sample
    for i in range(len(X)):
        try:
            if i < len(X)//2:  # Augment half the dataset
                features = augment_audio(X[i])
            else:
                # For non-augmented samples, ensure consistent shape
                features = X[i][:13]  # Take first 13 MFCCs
                delta = librosa.feature.delta(features)
                delta2 = librosa.feature.delta(features, order=2)
                features = np.vstack([features, delta, delta2])
            
            # Verify shape before adding
            if features.shape != (39, 130):
                print(f"Warning: Inconsistent shape {features.shape} for sample {i}")
                continue
                
            X_augmented.append(features)
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    # Convert to numpy array with explicit shape
    X_augmented = np.array(X_augmented)
    print(f"Final dataset shape: {X_augmented.shape}")
    
    # Ensure we have matching number of labels
    y = y[:len(X_augmented)]
    
    return X_augmented, y

# Load and prepare data
X, y = load_dataset()
print(f"Initial data shape: {X.shape}")  # Should be (samples, 39, 130)

# Convert labels to one-hot encoding
y = to_categorical(y)
print(f"Labels shape: {y.shape}")

# Reshape for model: (samples, time, features, channels)
X = np.transpose(X, (0, 2, 1))  # Now (samples, 130, 39)
X = X[..., np.newaxis]  # Add channel dimension
print(f"Final data shape: {X.shape}")  # Should be (samples, 130, 39, 1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
X_mean = np.mean(X_train, axis=(0, 1))  # Per-channel mean
X_std = np.std(X_train, axis=(0, 1))    # Per-channel std
X_train = (X_train - X_mean) / (X_std + 1e-8)
X_val = (X_val - X_mean) / (X_std + 1e-8)

# Create and compile model
model = create_model(input_shape=(130, 39))

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        # Warmup phase
        warmup_lr = self.initial_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        
        # Cosine decay phase
        progress = (tf.cast(step, tf.float32) - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + tf.cos(math.pi * tf.minimum(progress, 1.0)))
        cosine_lr = self.initial_lr * cosine_decay

        # Use warmup_lr for warmup phase, cosine_lr for rest
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    
    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }

batch_size = 16
epochs = 50

# Calculate steps
total_steps = (len(X_train) // batch_size) * epochs
warmup_steps = (len(X_train) // batch_size) * 5  # 5 epochs of warmup

# Create learning rate schedule
lr_schedule = WarmupCosineDecay(
    initial_lr=0.001,
    warmup_steps=warmup_steps,
    total_steps=total_steps
)

# Compile with gradient clipping
optimizer = Adam(lr_schedule, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Enhanced callbacks
callbacks = [
    EarlyStopping(
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

# Initialize tracker before training
tracker = ModelHistoryTracker()

# Define model configuration for tracking
model_config = {
    'batch_size': batch_size,
    'epochs': epochs,
    'initial_lr': lr_schedule.initial_lr,
    'model_params': model.count_params(),
    'regularization': {
        'l2': 0.01,
        'dropout': 0.5,
        'label_smoothing': 0.1
    }
}

# Define class names
class_names = ['blues', 'classical', 'country', 'disco', 
               'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)

# Save model
model.save("model/genre_classifier.keras")
print("\nModel saved to model/genre_classifier.keras")

# Generate visualizations
analyze_model_performance(model, history, X_train, y_train, X_val, y_val, class_names)

# Calculate final metrics
final_train_metrics = model.evaluate(X_train, y_train, verbose=0)
final_val_metrics = model.evaluate(X_val, y_val, verbose=0)

# Package metrics into dictionaries
train_metrics = {
    'loss': float(final_train_metrics[0]),
    'accuracy': float(final_train_metrics[1]),
    'auc': float(final_train_metrics[2])
}

val_metrics = {
    'loss': float(final_val_metrics[0]),
    'accuracy': float(final_val_metrics[1]),
    'auc': float(final_val_metrics[2])
}

# Save run history and plot comparisons
tracker.save_run(history, model_config, train_metrics, val_metrics)
tracker.print_summary()
tracker.plot_accuracy_comparison()

# Print final metrics
print("\nFinal Training Metrics:")
print(f"Loss: {train_metrics['loss']:.4f}")
print(f"Accuracy: {train_metrics['accuracy']:.4f}")
print(f"AUC: {train_metrics['auc']:.4f}")

print("\nFinal Validation Metrics:")
print(f"Loss: {val_metrics['loss']:.4f}")
print(f"Accuracy: {val_metrics['accuracy']:.4f}")
print(f"AUC: {val_metrics['auc']:.4f}")