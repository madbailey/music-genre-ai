import numpy as np
from src.data_loader import original_load_dataset
from src.augment import augment_audio
from src.model import create_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def mixup_data(X, y, alpha=0.2):
    """Performs mixup on the input data and their labels."""
    if alpha > 0:
        weights = np.random.beta(alpha, alpha, len(X))
        indices = np.random.permutation(len(X))
        
        X_weights = weights.reshape(len(X), 1, 1)
        X_mixed = X_weights * X + (1 - X_weights) * X[indices]
        
        y_weights = weights.reshape(len(X), 1)
        y_mixed = y_weights * y + (1 - y_weights) * y[indices]
        
        return X_mixed, y_mixed
    return X, y

def load_dataset():
    X, y = original_load_dataset()
    X_augmented = X.copy()
    for i in np.random.choice(len(X), size=len(X)//2, replace=False):
        X_augmented[i] = augment_audio(X_augmented[i])
    return X_augmented, y
# Load and prepare data
X, y = load_dataset()
X = np.transpose(X, (0, 2, 1))
y = to_categorical(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#normalize MFCC features to stabilize training
X_mean = np.mean(X_train, axis=(0, 1))  # Per-channel mean
X_std = np.std(X_train, axis=(0, 1))    # Per-channel std
X_train = (X_train - X_mean) / (X_std + 1e-8)
X_val = (X_val - X_mean) / (X_std + 1e-8)

# Create model
model = create_model(input_shape=X_train.shape[1:])
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)
model.compile(
    optimizer=Adam(lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', 'AUC']  # Added AUC back
)

# Training with mixup
batch_size = 16
epochs = 50
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

for epoch in range(epochs):
    # Apply mixup to training data
    X_mixed, y_mixed = mixup_data(X_train, y_train, alpha=0.4) # adjust 
    
    # Train for one epoch
    history = model.fit(
        X_mixed, y_mixed,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

# Save model
model.save("model/genre_classifier.keras")
print("\nModel saved to model/genre_classifier.keras")

# Print final metrics
final_train_loss, final_train_acc, final_train_auc = model.evaluate(X_train, y_train, verbose=0)
final_val_loss, final_val_acc, final_val_auc = model.evaluate(X_val, y_val, verbose=0)

print("\nFinal Training Metrics:")
print(f"Loss: {final_train_loss:.4f}")
print(f"Accuracy: {final_train_acc:.4f}")
print(f"AUC: {final_train_auc:.4f}")

print("\nFinal Validation Metrics:")
print(f"Loss: {final_val_loss:.4f}")
print(f"Accuracy: {final_val_acc:.4f}")
print(f"AUC: {final_val_auc:.4f}")