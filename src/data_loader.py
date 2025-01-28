import numpy as np
from pathlib import Path
import json
import librosa


def original_load_dataset():
    processed_data_path = Path("data/processed")

    # Load label map
    with open(processed_data_path / "label_map.json", "r") as f:
        label_map = json.load(f)

    X_mfccs = []
    X_spectral = []
    X_chroma = []
    y = []

    for genre_folder in processed_data_path.glob("*"):
        if genre_folder.is_dir():
            genre = genre_folder.name
            if genre in label_map:  # Skip label_map.json
                label = label_map[genre]
                for file_path in genre_folder.glob("*.npy"):
                    file_name = file_path.stem
                    
                    if file_name.endswith("_mfccs"):
                      
                      mfccs = np.load(file_path)
                      
                      #load associated features
                      spectral = np.load(genre_folder / f"{file_name.replace('_mfccs','')}_spectral.npy")
                      chroma = np.load(genre_folder / f"{file_name.replace('_mfccs','')}_chroma.npy")
                      
                      
                      X_mfccs.append(mfccs)
                      X_spectral.append(spectral)
                      X_chroma.append(chroma)
                      
                      y.append(label)

    
    return np.array(X_mfccs), np.array(X_spectral), np.array(X_chroma), np.array(y)