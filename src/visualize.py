import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf

def plot_training_history(history):
    """Plot training metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
def plot_learning_rate(history):
    """Plot learning rate changes."""
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
def plot_feature_importance(model, X_sample):
    """Plot feature importance using gradient-based analysis."""
    # Get gradients for a sample
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(X_sample)
        tape.watch(inputs)
        predictions = model(inputs)
        
    gradients = tape.gradient(predictions, inputs)
    importance = tf.reduce_mean(tf.abs(gradients), axis=[0, -1])
    
    plt.figure(figsize=(12, 4))
    plt.plot(importance)
    plt.title('Feature Importance across Time Steps')
    plt.xlabel('Time Step')
    plt.ylabel('Average Gradient Magnitude')
    plt.grid(True)
    
def analyze_model_performance(model, history, X_train, y_train, X_val, y_val, class_names):
    """Comprehensive model analysis with visualizations."""
    # 1. Plot training history
    plot_training_history(history)
    plt.savefig('training_history.png')
    plt.close()
    
    # 2. Plot learning rate changes
    plot_learning_rate(history)
    plt.savefig('learning_rate.png')
    plt.close()
    
    # 3. Get predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # 4. Plot confusion matrices
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(y_train, y_train_pred, class_names)
    plt.title('Training Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(y_val, y_val_pred, class_names)
    plt.title('Validation Confusion Matrix')
    plt.savefig('confusion_matrices.png')
    plt.close()
    
    # 5. Plot feature importance
    #plot_feature_importance(model, X_train[0]) #disable the feature importance for now
    
    # 6. Print classification report
    print("\nTraining Classification Report:")
    print(classification_report(y_train.argmax(axis=1), 
                              y_train_pred.argmax(axis=1), 
                              target_names=class_names))
    
    print("\nValidation Classification Report:")
    print(classification_report(y_val.argmax(axis=1), 
                              y_val_pred.argmax(axis=1), 
                              target_names=class_names))