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
    #load local audio and extract mfccs
    audio, sr = librosa.load(file_path, sr=SR)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC).T #(time, n_mfcc)

    if mfccs.shape[0] < max_length:
        pad = max_length - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:max_length, :]

    return mfccs 



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