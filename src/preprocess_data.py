import librosa
import numpy as np
from pathlib import Path
import json

# Configuration
SR = 22050
DURATION = 30  # in seconds
N_MFCC = 13
MAX_LENGTH = 130


def process_audio(file_path, max_length=MAX_LENGTH):
    # Load local audio and extract features
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

    # Extract individual features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    # Resize features to match dimensions
    mfccs = librosa.util.fix_length(mfccs, size=max_length, axis=1)
    spectral_centroids = librosa.util.fix_length(
        spectral_centroids, size=max_length, axis=1
    )
    chroma = librosa.util.fix_length(chroma, size=max_length, axis=1)
    
    return mfccs, spectral_centroids, chroma  # Return as separate arrays


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
                mfccs, spectral_centroids, chroma = process_audio(file)
                
                # Save all feature components
                np.save(save_dir / f"{file.stem}_mfccs.npy", mfccs)
                np.save(save_dir / f"{file.stem}_spectral.npy", spectral_centroids)
                np.save(save_dir / f"{file.stem}_chroma.npy", chroma)
                
                print(f"Saved all features to {save_dir}")
                total_files += 1
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue  # Skip corrupted files

    print(f"Processed {total_files} files successfully")

if __name__ == "__main__":
    save_processed_data()