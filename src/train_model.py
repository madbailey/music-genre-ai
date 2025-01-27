import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset 
from src.model import create_model  
from tensorflow.keras.callbacks import EarlyStopping

# Load data
X, y = load_dataset()
print(f"Raw data shape: {X.shape}")  # Should be (1000, 130, 13)


X = X[..., np.newaxis]  # New shape: (1000, 130, 13, 1)
print(f"With channel dim: {X.shape}")  # Should be (1000, 130, 13, 1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

# Create model
model = create_model(input_shape=X_train.shape[1:])

# Train with early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# Save model
model.save("model/genre_classifier.keras")
print("\nModel saved to model/genre_classifier.keras")