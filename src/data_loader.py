import numpy as np
from pathlib import Path

def original_load_dataset():
    processed_data_path = Path("data/processed")
    
    # Load label map
    import json
    with open(processed_data_path / "label_map.json", "r") as f:
        label_map = json.load(f)
    
    X = []
    y = []
    
    for genre_folder in processed_data_path.glob("*"):
        if genre_folder.is_dir():
            genre = genre_folder.name
            if genre in label_map:  # Skip label_map.json
                label = label_map[genre]
                for npy_file in genre_folder.glob("*.npy"):
                    features = np.load(npy_file)
                    X.append(features)
                    y.append(label)
    
    return np.array(X), np.array(y)