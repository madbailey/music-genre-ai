from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def create_model(input_shape, num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First Conv Block
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),
        
        # Second Conv Block
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        
        # Third Conv Block
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model  # Remove compile from here since we'll compile in train_model.py