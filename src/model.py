# Updated model.py
import tensorflow as tf
from tensorflow.keras import layers, models

# Create and compile model
def create_residual_block(x, filters, kernel_size=3):
    """Create a residual block with skip connection"""
    skip = x
    
    # Main path
    x = layers.Conv1D(filters, kernel_size, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.3)(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    if skip.shape[-1] != filters:
        skip = layers.Conv1D(filters, 1, padding='same')(skip)
    
    # Add skip connection
    x = layers.Add()([x, skip])
    x = layers.ReLU()(x)
    return x

def create_model(input_shape_mfcc, input_shape_spectral, input_shape_chroma, num_classes=10):
    #input layers for features
    mfcc_inputs = layers.Input(shape=input_shape_mfcc)
    spectral_inputs = layers.Input(shape=input_shape_spectral)
    chroma_inputs = layers.Input(shape=input_shape_chroma)
    
    # Initial convolution for mfccs
    mfcc_x = layers.Conv1D(32, 7, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.02))(mfcc_inputs)
    mfcc_x = layers.BatchNormalization()(mfcc_x)
    mfcc_x = layers.ReLU()(mfcc_x)
    mfcc_x = layers.MaxPooling1D(2)(mfcc_x)
    
    # Initial convolution for spectral features
    spectral_x = layers.Conv1D(32, 7, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.02))(spectral_inputs)
    spectral_x = layers.BatchNormalization()(spectral_x)
    spectral_x = layers.ReLU()(spectral_x)
    spectral_x = layers.MaxPooling1D(2)(spectral_x)
    
    # Initial convolution for chroma features
    chroma_x = layers.Conv1D(32, 7, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.02))(chroma_inputs)
    chroma_x = layers.BatchNormalization()(chroma_x)
    chroma_x = layers.ReLU()(chroma_x)
    chroma_x = layers.MaxPooling1D(2)(chroma_x)
    
    # Residual blocks for mfcc
    mfcc_x = create_residual_block(mfcc_x, 32)
    mfcc_x = layers.MaxPooling1D(2)(mfcc_x)
    
    mfcc_x = create_residual_block(mfcc_x, 64)
    mfcc_x = layers.MaxPooling1D(2)(mfcc_x)
    
    # Residual blocks for spectral features
    spectral_x = create_residual_block(spectral_x, 32)
    spectral_x = layers.MaxPooling1D(2)(spectral_x)
    
    spectral_x = create_residual_block(spectral_x, 64)
    spectral_x = layers.MaxPooling1D(2)(spectral_x)
    
    # Residual blocks for chroma features
    chroma_x = create_residual_block(chroma_x, 32)
    chroma_x = layers.MaxPooling1D(2)(chroma_x)
    
    chroma_x = create_residual_block(chroma_x, 64)
    chroma_x = layers.MaxPooling1D(2)(chroma_x)
    
    # Global pooling
    mfcc_x = layers.GlobalAveragePooling1D()(mfcc_x)
    spectral_x = layers.GlobalAveragePooling1D()(spectral_x)
    chroma_x = layers.GlobalAveragePooling1D()(chroma_x)
    
    # Concatenate the outputs from all paths
    x = layers.concatenate([mfcc_x, spectral_x, chroma_x])
    
    # Classification head
    x = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model([mfcc_inputs, spectral_inputs, chroma_inputs], outputs)