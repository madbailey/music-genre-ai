import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from src.model_progress import ModelHistoryTracker

class ModelTrainer:
    def __init__(self, input_shapes, num_classes):
        self.input_shape_mfcc = input_shapes['mfcc']
        self.input_shape_spectral = input_shapes['spectral']
        self.input_shape_chroma = input_shapes['chroma']
        self.num_classes = num_classes
        self.tracker = ModelHistoryTracker()


    def _add_attention_mechanism(self, x):
        """Custom attention mechanism to replace GlobalAttention"""
        # Create attention layer
        attention_weights = layers.Dense(1, activation='tanh')(x)  # (batch_size, time_steps, 1)
        attention_weights = layers.Softmax(axis=1)(attention_weights)  # Normalize over time steps
        
        # Apply attention weights
        context_vector = layers.Multiply()([x, attention_weights])
        context_vector = layers.GlobalAveragePooling1D()(context_vector)
        
        return context_vector

    def build_model(self, hp):
        """Hypermodel builder for Keras Tuner"""
        # Input layers for features
        mfcc_inputs = layers.Input(shape=self.input_shape_mfcc)
        spectral_inputs = layers.Input(shape=self.input_shape_spectral)
        chroma_inputs = layers.Input(shape=self.input_shape_chroma)
        
        # Initial convolution for mfccs
        mfcc_x = layers.Conv1D(
            hp.Int('mfcc_conv1_filters', min_value=32, max_value=256, step=32), 
            hp.Choice('mfcc_kernel_size', values=[3, 5, 7]), 
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2_reg', min_value=0.001, max_value=0.1, sampling='log')))(mfcc_inputs)
        mfcc_x = layers.BatchNormalization()(mfcc_x)
        mfcc_x = layers.ReLU()(mfcc_x)
        mfcc_x = layers.MaxPooling1D(2)(mfcc_x)
        
        # Initial convolution for spectral features
        spectral_x = layers.Conv1D(
            hp.Int('spectral_conv1_filters', min_value=32, max_value=256, step=32),
            hp.Choice('spectral_kernel_size', values=[3, 5, 7]),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2_reg', min_value=0.001, max_value=0.1, sampling='log')))(spectral_inputs)
        spectral_x = layers.BatchNormalization()(spectral_x)
        spectral_x = layers.ReLU()(spectral_x)
        spectral_x = layers.MaxPooling1D(2)(spectral_x)
        
        # Initial convolution for chroma features
        chroma_x = layers.Conv1D(
            hp.Int('chroma_conv1_filters', min_value=32, max_value=256, step=32),
            hp.Choice('chroma_kernel_size', values=[3, 5, 7]),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2_reg', min_value=0.001, max_value=0.1, sampling='log')))(chroma_inputs)
        chroma_x = layers.BatchNormalization()(chroma_x)
        chroma_x = layers.ReLU()(chroma_x)
        chroma_x = layers.MaxPooling1D(2)(chroma_x)

        # Add attention mechanism if specified
        if hp.Boolean('use_attention'):
            mfcc_x = self._add_attention_mechanism(mfcc_x)
            spectral_x = self._add_attention_mechanism(spectral_x)
            chroma_x = self._add_attention_mechanism(chroma_x)
            
            # If using attention, we don't need residual blocks as attention has processed the sequences
            mfcc_features = mfcc_x
            spectral_features = spectral_x
            chroma_features = chroma_x
        else:
            # Residual blocks if not using attention
            for i in range(hp.Int('num_res_blocks', 2, 4)):
                mfcc_x = self._create_residual_block(
                    mfcc_x, 
                    hp.Int(f'mfcc_res{i}_filters', min_value=64, max_value=512, step=64),
                    hp.Choice(f'mfcc_res{i}_kernel', values=[3, 5])
                )
                spectral_x = self._create_residual_block(
                    spectral_x,
                    hp.Int(f'spectral_res{i}_filters', min_value=64, max_value=512, step=64),
                    hp.Choice(f'spectral_res{i}_kernel', values=[3, 5])
                )
                chroma_x = self._create_residual_block(
                    chroma_x,
                    hp.Int(f'chroma_res{i}_filters', min_value=64, max_value=512, step=64),
                    hp.Choice(f'chroma_res{i}_kernel', values=[3, 5])
                )
                
                # Add pooling after each residual block
                mfcc_x = layers.MaxPooling1D(2)(mfcc_x)
                spectral_x = layers.MaxPooling1D(2)(spectral_x)
                chroma_x = layers.MaxPooling1D(2)(chroma_x)
                
            # Global pooling
            mfcc_features = layers.GlobalAveragePooling1D()(mfcc_x)
            spectral_features = layers.GlobalAveragePooling1D()(spectral_x)
            chroma_features = layers.GlobalAveragePooling1D()(chroma_x)
        
        # Concatenate features
        x = layers.concatenate([mfcc_features, spectral_features, chroma_features])
        
        # Dense layers with residual connections
        for i in range(hp.Int('num_dense_layers', 1, 3)):
            skip = x
            units = hp.Int(f'dense_units_{i}', min_value=64, max_value=512, step=64)
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5))(x)
            
            # Add residual if shapes match
            if skip.shape[-1] == units:
                x = layers.Add()([x, skip])
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = models.Model([mfcc_inputs, spectral_inputs, chroma_inputs], outputs)
        
        # Compile with advanced settings
        optimizer = Adam(
            hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log'),
            clipnorm=hp.Float('clipnorm', min_value=0.1, max_value=1.0)
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=hp.Float('label_smoothing', min_value=0.0, max_value=0.2)
            ),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        
        return model

    def _create_residual_block(self, x, filters, kernel_size):
        """Enhanced residual block with additional normalization and activation"""
        skip = x
        
        # First conv block
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Second conv block
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Skip connection with projection if needed
        if skip.shape[-1] != filters:
            skip = layers.Conv1D(filters, 1, padding='same')(skip)
            skip = layers.BatchNormalization()(skip)
        
        x = layers.Add()([x, skip])
        x = layers.ReLU()(x)
        
        return x

    def _add_attention_layer(self, x):
        """Add self-attention mechanism"""
        # Create query, key, value transformations
        query = layers.Dense(x.shape[-1])(x)
        key = layers.Dense(x.shape[-1])(x)
        value = layers.Dense(x.shape[-1])(x)
        
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(x.shape[-1], tf.float32))
        attention_weights = tf.nn.softmax(scores)
        
        # Apply attention
        attended = tf.matmul(attention_weights, value)
        return layers.Add()([x, attended])  # Residual connection