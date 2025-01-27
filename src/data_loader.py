import numpy as np
import json
from pathlib import Path

def load_dataset(processed_dir="data/processed"):
    processed_dir = Path(processed_dir)
    X, y = [], []
    
    # Load label map
    with open(processed_dir / "label_map.json", "r") as f:
        label_map = json.load(f)
    
    # Load all .npy files
    for genre, label in label_map.items():
        genre_dir = processed_dir / genre
        for file in genre_dir.glob("*.npy"):
            X.append(np.load(file))
            y.append(label)
    
    return np.array(X), np.array(y)

# Test it
if __name__ == "__main__":
    X, y = load_dataset()
    print(f"Dataset shapes: X={X.shape}, y={y.shape}")