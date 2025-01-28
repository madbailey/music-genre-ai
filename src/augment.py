import librosa
import numpy as np

def augment_audio(mfccs, sr=22050):
    # Random scaling
    scale_factor = np.random.uniform(0.6, 1.4)  # Wider range
    mfccs = mfccs * scale_factor
    
    # Add noise
    noise = np.random.normal(0, 0.01, mfccs.shape)  # Increased noise
    mfccs = mfccs + noise
    
    # Time masking
    time_steps = mfccs.shape[1]
    max_mask_time = int(time_steps * 0.1)  # Mask up to 10% of time steps
    if max_mask_time > 0:
        t0 = np.random.randint(0, time_steps - max_mask_time)
        mfccs[:, t0:t0+max_mask_time] = 0
    
    # Frequency masking
    freq_bins = mfccs.shape[0]
    max_mask_freq = int(freq_bins * 0.1)  # Mask up to 10% of freq bins
    if max_mask_freq > 0:
        f0 = np.random.randint(0, freq_bins - max_mask_freq)
        mfccs[f0:f0+max_mask_freq, :] = 0
    
    return mfccs