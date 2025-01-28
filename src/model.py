# Updated model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_residual_block(x, filters, kernel_size=3):
    """Create a residual block with skip connection"""
    skip = x
    
    # Main path
    x = layers.Conv1D(filters, kernel_size, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.2)(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    if skip.shape[-1] != filters:
        skip = layers.Conv1D(filters, 1, padding='same')(skip)
    
    # Add skip connection
    x = layers.Add()([x, skip])
    x = layers.ReLU()(x)
    return x

def create_model(input_shape, num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv1D(32, 7, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Residual blocks
    x = create_residual_block(x, 32)
    x = layers.MaxPooling1D(2)(x)
    
    x = create_residual_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)