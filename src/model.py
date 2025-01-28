from tensorflow.keras import layers, models

# In model.py
def create_model(input_shape, num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Reduced filters
        layers.Conv1D(32, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),  # Increased dropout
        
        # Removed second conv block
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.6),  # Added
        layers.Dense(num_classes, activation='softmax')
    ])
    return model