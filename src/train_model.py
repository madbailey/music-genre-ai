import math
import numpy as np
import librosa
from src.data_loader import original_load_dataset
from src.augment import augment_audio, time_warp
from src.model import create_model, create_residual_block
from src.visualize import analyze_model_performance
from src.model_progress import ModelHistoryTracker
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
import os
import keras_tuner as kt

def mixup_data(X_mfccs, X_spectral, X_chroma, y, alpha=0.2):
    """Performs mixup on the input data and their labels.
    
    Args:
        X: Input data, shape (samples, time_steps, features, channels)
        y: One-hot encoded labels, shape (samples, num_classes)
        alpha: Mixup interpolation strength
    
    Returns:
        Tuple of mixed input data and labels
    """
    if alpha <= 0:
        return X_mfccs, X_spectral, X_chroma, y
        
    # Validate inputs
    if len(X_mfccs.shape) != 3:
        raise ValueError(f"Expected 3D input tensor, got shape {X_mfccs.shape}")
    if len(X_spectral.shape) != 3:
        raise ValueError(f"Expected 3D input tensor, got shape {X_spectral.shape}")
    if len(X_chroma.shape) != 3:
        raise ValueError(f"Expected 3D input tensor, got shape {X_chroma.shape}")
    if len(y.shape) != 2:
        raise ValueError(f"Expected 2D one-hot encoded labels, got shape {y.shape}")
    
    # Generate mixup weights
    weights = np.random.beta(alpha, alpha, len(X_mfccs))
    indices = np.random.permutation(len(X_mfccs))
    
    # Reshape weights for broadcasting
    X_weights = weights.reshape(len(X_mfccs), 1, 1)
    y_weights = weights.reshape(len(X_mfccs), 1)
    
    # Perform mixup
    X_mfccs_mixed = X_weights * X_mfccs + (1 - X_weights) * X_mfccs[indices]
    X_spectral_mixed = X_weights * X_spectral + (1 - X_weights) * X_spectral[indices]
    X_chroma_mixed = X_weights * X_chroma + (1 - X_weights) * X_chroma[indices]
    y_mixed = y_weights * y + (1 - y_weights) * y[indices]
    
    return X_mfccs_mixed, X_spectral_mixed, X_chroma_mixed, y_mixed

def load_dataset():
    """
    Load and augment dataset with consistent shapes
    Returns:
        X: numpy array of shape (samples, 39, 130)
        y: numpy array of shape (samples,)
    """
    X_mfccs, X_spectral, X_chroma, y = original_load_dataset()
    
    X_mfccs_augmented = []
    X_spectral_augmented = []
    X_chroma_augmented = []
    
    # Process each sample
    for i in range(len(X_mfccs)):
      try:
        #augment mfccs
        mfccs = augment_audio(X_mfccs[i])
        mfccs = time_warp(mfccs)
        
        # Extract delta features from augmented mfccs
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        features = np.vstack([mfccs, delta, delta2])
                
        # Ensure consistent shape for mfccs
        if features.shape != (39, 130):
          print(f"Warning: Inconsistent shape {features.shape} for mfcc sample {i}")
          continue
        X_mfccs_augmented.append(features)
        
        
        #Process spectral and chroma features without augmentation
        spectral = X_spectral[i]
        if spectral.shape != (1, 130):
           print(f"Warning: Inconsistent shape {spectral.shape} for spectral sample {i}")
           continue
        X_spectral_augmented.append(spectral)
        
        chroma = X_chroma[i]
        if chroma.shape != (12,130):
            print(f"Warning: Inconsistent shape {chroma.shape} for chroma sample {i}")
            continue
        X_chroma_augmented.append(chroma)
            
      except Exception as e:
          print(f"Error processing sample {i}: {str(e)}")
          continue
          
    # Convert to numpy array with explicit shape
    X_mfccs_augmented = np.array(X_mfccs_augmented)
    X_spectral_augmented = np.array(X_spectral_augmented)
    X_chroma_augmented = np.array(X_chroma_augmented)
    
    print(f"Final MFCC shape: {X_mfccs_augmented.shape}")
    print(f"Final Spectral shape: {X_spectral_augmented.shape}")
    print(f"Final Chroma shape: {X_chroma_augmented.shape}")
    
    # Ensure we have matching number of labels
    y = y[:len(X_mfccs_augmented)]
    
    return X_mfccs_augmented, X_spectral_augmented, X_chroma_augmented, y

# Load and prepare data
X_mfccs, X_spectral, X_chroma, y = load_dataset()

# Split data
X_mfccs_train, X_mfccs_val, X_spectral_train, X_spectral_val, X_chroma_train, X_chroma_val, y_train, y_val = train_test_split(
    X_mfccs, X_spectral, X_chroma, y, test_size=0.2, random_state=42
)

#one hot encode y before mixup
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print(f"Labels shape: {y_train.shape}")

#mixup
X_mfccs_train, X_spectral_train, X_chroma_train, y_train = mixup_data(X_mfccs_train, X_spectral_train, X_chroma_train, y_train)

# Reshape for model: (samples, time, features, channels)
X_mfccs_train = np.transpose(X_mfccs_train, (0, 2, 1))  # Now (samples, 130, 39)
X_mfccs_train = X_mfccs_train[..., np.newaxis]  # Add channel dimension
print(f"Final MFCC data shape: {X_mfccs_train.shape}") 

X_spectral_train = np.transpose(X_spectral_train, (0, 2, 1))
X_spectral_train = X_spectral_train[..., np.newaxis]
print(f"Final spectral data shape: {X_spectral_train.shape}")

X_chroma_train = np.transpose(X_chroma_train, (0, 2, 1))
X_chroma_train = X_chroma_train[..., np.newaxis]
print(f"Final chroma data shape: {X_chroma_train.shape}")
    
X_mfccs_val = np.transpose(X_mfccs_val, (0, 2, 1))  # Now (samples, 130, 39)
X_mfccs_val = X_mfccs_val[..., np.newaxis]  # Add channel dimension

X_spectral_val = np.transpose(X_spectral_val, (0, 2, 1))
X_spectral_val = X_spectral_val[..., np.newaxis]

X_chroma_val = np.transpose(X_chroma_val, (0, 2, 1))
X_chroma_val = X_chroma_val[..., np.newaxis]

# Normalize features individually
X_mfccs_mean = np.mean(X_mfccs_train, axis=(0, 1))  # Per-channel mean
X_mfccs_std = np.std(X_mfccs_train, axis=(0, 1))    # Per-channel std
X_mfccs_train = (X_mfccs_train - X_mfccs_mean) / (X_mfccs_std + 1e-8)
X_mfccs_val = (X_mfccs_val - X_mfccs_mean) / (X_mfccs_std + 1e-8)

X_spectral_mean = np.mean(X_spectral_train, axis=(0, 1))
X_spectral_std = np.std(X_spectral_train, axis=(0,1))
X_spectral_train = (X_spectral_train - X_spectral_mean) / (X_spectral_std + 1e-8)
X_spectral_val = (X_spectral_val - X_spectral_mean) / (X_spectral_std + 1e-8)

X_chroma_mean = np.mean(X_chroma_train, axis=(0, 1))
X_chroma_std = np.std(X_chroma_train, axis=(0, 1))
X_chroma_train = (X_chroma_train - X_chroma_mean) / (X_chroma_std + 1e-8)
X_chroma_val = (X_chroma_val - X_chroma_mean) / (X_chroma_std + 1e-8)

def build_model(hp):
    input_shape_mfcc=(130,39)
    input_shape_spectral=(130, 1)
    input_shape_chroma=(130,12)
    num_classes = y_train.shape[1]  # Define num_classes based on the shape of y_train
    
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
    
    #input layers for features
    mfcc_inputs = layers.Input(shape=input_shape_mfcc)
    spectral_inputs = layers.Input(shape=input_shape_spectral)
    chroma_inputs = layers.Input(shape=input_shape_chroma)
    
    # Initial convolution for mfccs
    mfcc_x = layers.Conv1D(
        hp.Int('mfcc_conv1_filters', min_value=32, max_value=128, step=32), 7, padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_reg', min_value=0.01, max_value=0.03, step=0.01)))(mfcc_inputs)
    mfcc_x = layers.BatchNormalization()(mfcc_x)
    mfcc_x = layers.ReLU()(mfcc_x)
    mfcc_x = layers.MaxPooling1D(2)(mfcc_x)
    
    # Initial convolution for spectral features
    spectral_x = layers.Conv1D(
        hp.Int('spectral_conv1_filters', min_value=32, max_value=128, step=32), 7, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_reg', min_value=0.01, max_value=0.03, step=0.01)))(spectral_inputs)
    spectral_x = layers.BatchNormalization()(spectral_x)
    spectral_x = layers.ReLU()(spectral_x)
    spectral_x = layers.MaxPooling1D(2)(spectral_x)
    
    # Initial convolution for chroma features
    chroma_x = layers.Conv1D(
         hp.Int('chroma_conv1_filters', min_value=32, max_value=128, step=32), 7, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_reg', min_value=0.01, max_value=0.03, step=0.01)))(chroma_inputs)
    chroma_x = layers.BatchNormalization()(chroma_x)
    chroma_x = layers.ReLU()(chroma_x)
    chroma_x = layers.MaxPooling1D(2)(chroma_x)
    
    # Residual blocks for mfcc
    mfcc_x = create_residual_block(mfcc_x, hp.Int('mfcc_res1_filters', min_value=32, max_value=128, step=32), kernel_size=3)
    mfcc_x = layers.MaxPooling1D(2)(mfcc_x)
    
    mfcc_x = create_residual_block(mfcc_x, hp.Int('mfcc_res2_filters', min_value=64, max_value=256, step=32), kernel_size=3)
    mfcc_x = layers.MaxPooling1D(2)(mfcc_x)
    
    # Residual blocks for spectral features
    spectral_x = create_residual_block(spectral_x, hp.Int('spectral_res1_filters', min_value=32, max_value=128, step=32), kernel_size = 3)
    spectral_x = layers.MaxPooling1D(2)(spectral_x)
    
    spectral_x = create_residual_block(spectral_x, hp.Int('spectral_res2_filters', min_value=64, max_value=256, step=32), kernel_size = 3)
    spectral_x = layers.MaxPooling1D(2)(spectral_x)
    
    # Residual blocks for chroma features
    chroma_x = create_residual_block(chroma_x, hp.Int('chroma_res1_filters', min_value=32, max_value=128, step=32), kernel_size = 3)
    chroma_x = layers.MaxPooling1D(2)(chroma_x)
    
    chroma_x = create_residual_block(chroma_x, hp.Int('chroma_res2_filters', min_value=64, max_value=256, step=32), kernel_size = 3)
    chroma_x = layers.MaxPooling1D(2)(chroma_x)
    
    # Global pooling
    mfcc_x = layers.GlobalAveragePooling1D()(mfcc_x)
    spectral_x = layers.GlobalAveragePooling1D()(spectral_x)
    chroma_x = layers.GlobalAveragePooling1D()(chroma_x)
    
    # Concatenate the outputs from all paths
    x = layers.concatenate([mfcc_x, spectral_x, chroma_x])
    
    # Classification head
    x = layers.Dense(hp.Int('dense_units_1', min_value=32, max_value=128, step = 32), kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_reg', min_value=0.01, max_value=0.03, step=0.01)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(hp.Float('dropout_rate', min_value = 0.3, max_value=0.7, step=0.1))(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model([mfcc_inputs, spectral_inputs, chroma_inputs], outputs)
    
    # Define learning rate schedule
    lr_schedule = WarmupCosineDecay(
        initial_lr=hp.Float('initial_lr', min_value=0.0001, max_value=0.001, step = 0.0001),
        warmup_steps=(len(X_mfccs_train) // 32) * 5,
        total_steps=(len(X_mfccs_train) // 32) * 75
    )

    # Compile with gradient clipping
    optimizer = Adam(lr_schedule, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=hp.Float("label_smoothing", min_value=0.01, max_value=0.1, step = 0.01)
        ),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model
    
tuner = kt.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        directory='tuning_dir',
        project_name='audio_hyperparam_tune'
    )

batch_size = 32
epochs = 75

tuner.search([X_mfccs_train, X_spectral_train, X_chroma_train], y_train, 
                validation_data=([X_mfccs_val, X_spectral_val, X_chroma_val], y_val),
                epochs=epochs,
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
                     ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath = os.path.join("model","checkpoints", "genre_model_{epoch:02d}-{val_accuracy:.2f}.keras"),
                        monitor="val_accuracy",
                        save_best_only = True,
                        save_freq = "epoch"
                        )
                ], verbose = 1)

best_model = tuner.get_best_models(num_models=1)[0]
model = best_model
model.save("model/genre_classifier.keras")
print("\nModel saved to model/genre_classifier.keras")

# Generate visualizations
analyze_model_performance(model, history, [X_mfccs_train, X_spectral_train, X_chroma_train], y_train, [X_mfccs_val, X_spectral_val, X_chroma_val], y_val, class_names)

# Calculate final metrics
final_train_metrics = model.evaluate([X_mfccs_train, X_spectral_train, X_chroma_train], y_train, verbose=0)
final_val_metrics = model.evaluate([X_mfccs_val, X_spectral_val, X_chroma_val], y_val, verbose=0)

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