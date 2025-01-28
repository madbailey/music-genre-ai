import librosa
import numpy as np 
import os
from pathlib import Path
import json

#configuration
SR = 22050
DURATION = 30 #in seconds
N_MFCC = 13
MAX_LENGTH = 130

def process_audio(file_path, max_length=MAX_LENGTH):
    #load local audio and extract features
    audio, sr = librosa.load(file_path, sr=SR)
    
    # Add validation
    if len(audio) < SR:  # Less than 1 second
        raise ValueError("Audio file too short")
        
    # Ensure consistent length
    if len(audio) > SR * DURATION:
        audio = audio[:SR * DURATION]
    elif len(audio) < SR * DURATION:
        padding = SR * DURATION - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    
    # Extract multiple features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Normalize each feature
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-6)
    spectral_centroids = (spectral_centroids - np.mean(spectral_centroids)) / (np.std(spectral_centroids) + 1e-6)
    chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-6)
    
    # Resize features to match dimensions
    target_length = 130  # Your MAX_LENGTH
    mfccs = librosa.util.fix_length(mfccs, size=target_length, axis=1)
    spectral_centroids = librosa.util.fix_length(spectral_centroids, size=target_length, axis=1)
    chroma = librosa.util.fix_length(chroma, size=target_length, axis=1)
    
    # Stack features
    features = np.vstack([mfccs,  # 13 features
                         spectral_centroids,  # 1 feature
                         chroma])  # 12 features
    
    return features  # Shape will be (26, 130)


def save_processed_data():
    raw_data_path = Path("data/raw")
    processed_data_path = Path("data/processed")
    
    print(f"Raw data path exists: {raw_data_path.exists()}")
    print(f"Raw data path absolute: {raw_data_path.absolute()}")
    
    # Create genre-to-index mapping
    genres = sorted([d.name for d in raw_data_path.iterdir() if d.is_dir()])
    print(f"Found genres: {genres}")
    
    label_map = {genre: i for i, genre in enumerate(genres)}
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created processed directory at: {processed_data_path.absolute()}")
    
    # Save label map
    with open(processed_data_path / "label_map.json", "w") as f:
        json.dump(label_map, f)
    
    # Process all files
    total_files = 0
    for genre in genres:
        genre_path = raw_data_path / genre
        save_dir = processed_data_path / genre
        save_dir.mkdir(exist_ok=True)
        
        wav_files = list(genre_path.glob("*.wav"))
        print(f"Found {len(wav_files)} WAV files in {genre}")
        
        for file in wav_files:
            try:
                print(f"Processing {file}")
                mfccs = process_audio(file)
                save_path = save_dir / f"{file.stem}.npy"
                np.save(save_path, mfccs)
                print(f"Saved to {save_path}")
                total_files += 1
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue  # Skip corrupted files
            
    print(f"Processed {total_files} files successfully")

if __name__ == "__main__":
    save_processed_data()