import librosa
import numpy as np

def augment_audio(mfccs, sr=22050):
    """
    Augment MFCC features directly instead of raw audio
    Input shape: (13, 130)
    Output shape: (13, 130)
    """
    # Random scaling
    scale_factor = np.random.uniform(0.8, 1.2)
    mfccs = mfccs * scale_factor
    
    # Add small random noise
    noise = np.random.normal(0, 0.001, mfccs.shape)
    mfccs = mfccs + noise
    
    return mfccs