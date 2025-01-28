import librosa
import numpy as np

def augment_audio(mfccs):
    """Enhanced audio augmentation with multiple techniques"""
    if len(mfccs.shape) != 2:
        raise ValueError(f"Expected 2D input, got shape {mfccs.shape}")
    
    # Take base MFCCs
    
    # Random scaling with wider range
    scale_factor = np.random.uniform(0.8, 1.05)
    mfccs = mfccs * scale_factor
    
    # SpecAugment-like frequency masking
    num_freq_masks = np.random.randint(1, 3)
    for _ in range(num_freq_masks):
        freq_mask_param = np.random.randint(1, 2)
        freq_mask_idx = np.random.randint(0, mfccs.shape[0] - freq_mask_param)
        mfccs[freq_mask_idx:freq_mask_idx + freq_mask_param] = 0
    
    # SpecAugment-like time masking
    num_time_masks = np.random.randint(1, 3)
    for _ in range(num_time_masks):
        time_mask_param = np.random.randint(3, 7)
        time_mask_idx = np.random.randint(0, mfccs.shape[1] - time_mask_param)
        mfccs[:, time_mask_idx:time_mask_idx + time_mask_param] = 0
        
    return mfccs

def time_warp(mfccs, max_warp=5):
    """Simulate tempo changes via MFCC rolling"""
    warp_steps = np.random.randint(-max_warp, max_warp)
    return np.roll(mfccs, warp_steps, axis=1)