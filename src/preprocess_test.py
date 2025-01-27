import librosa
import numpy as np 

def test_audio_processing():
    audio_path ="data/raw/blues.00000.wav"
    signal, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    print(f"Success! MFCC shape: {mfccs.shape}")
    assert mfccs.shape == (13, 1293), "Unexpected MFCC shape"

if __name__ == "__main__":
    test_audio_processing()