from tensorflow.keras import layers, models

def create_model(input_shape, num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First Conv Block
        layers.Conv2D(24, (5,5), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        
        # Second Conv Block
        layers.Conv2D(48, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model