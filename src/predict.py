import numpy as np
import json
from tensorflow.keras.models import load_model
from src.data_loader import load_dataset 
from src.preprocess_data import process_audio
import librosa

def predict_genre(file_path):
    # Load model and label map
    model = load_model("model/genre_classifier.keras")
    with open("data/processed/label_map.json", "r") as f:
        label_map = json.load(f)
    id_to_genre = {v: k for k, v in label_map.items()}
    
    # Preprocess new audio
    mfccs = process_audio(file_path)  # Use your existing function
    mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
    
    # Predict
    print(f"Input shape: {mfccs.shape}")  # Should be (1, 130, 13, 1)
    probs = model.predict(mfccs)[0]
    predicted_id = np.argmax(probs)
    return id_to_genre[predicted_id], probs

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_audio.wav>")
        sys.exit(1)
    
    genre, confidence = predict_genre(sys.argv[1])
    print(f"Predicted genre: {genre} (confidence: {confidence.max():.2%})")