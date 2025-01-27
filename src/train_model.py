import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset 
from src.model import create_model  
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical  # Add this import
import matplotlib.pyplot as plt


# Load data
X, y = load_dataset()
print(f"Raw data shape: {X.shape}")  # Should be (1000, 130, 13)


X = np.transpose(X, (0, 2, 1))
X = X[..., np.newaxis]  # New shape: (1000, 130, 13, 1)
print(f"With channel dim: {X.shape}")  # Should be (1000, 130, 13, 1)

y = to_categorical(y)  # Add this line
print(f"Labels shape: {y.shape}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

# Create callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Create and train model
model = create_model(input_shape=X_train.shape[1:])
model.compile(  # Recompile with categorical_crossentropy
    optimizer='adam',
    loss='categorical_crossentropy',  # Changed from sparse_categorical_crossentropy
    metrics=['accuracy', 'AUC']
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# Save model
model.save("model/genre_classifier.keras")
print("\nModel saved to model/genre_classifier.keras")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model/training_history.png')
plt.close()

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