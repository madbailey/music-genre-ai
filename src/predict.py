import numpy as np
import json
from tensorflow.keras.models import load_model
from src.preprocess_data import process_audio
from pathlib import Path
import os
import librosa


def predict_genre(file_path):
    # Load model and label map
    model_dir = Path("model/checkpoints")
    model_paths = sorted(list(model_dir.glob("*.keras")))
    print(f"Found model paths: {model_paths}") #verify models loaded
    models = [load_model(str(path)) for path in model_paths]
    if not models:
      print("No models in checkpoint path, loading base model")
      models = [load_model("model/genre_classifier.keras")]
    
    with open("data/processed/label_map.json", "r") as f:
        label_map = json.load(f)
    id_to_genre = {v: k for k, v in label_map.items()}
    
    # Preprocess new audio
    mfccs, spectral, chroma = process_audio(file_path)  # Use your existing function
    mfccs = np.transpose(mfccs, (1, 0))
    mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
    spectral = np.transpose(spectral, (1, 0))
    spectral = spectral[np.newaxis, ..., np.newaxis]
    chroma = np.transpose(chroma, (1, 0))
    chroma = chroma[np.newaxis, ..., np.newaxis]
    
    # Predict using all models and average
    all_probs = []
    for model in models:
      print(f"Loaded model:{model}")
      all_probs.append(model.predict([mfccs, spectral, chroma])[0])
    probs = np.mean(all_probs, axis=0)
    predicted_id = np.argmax(probs)
    return id_to_genre[predicted_id], probs

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_audio.wav>")
        sys.exit(1)
    
    genre, confidence = predict_genre(sys.argv[1])
    print(f"Predicted genre: {genre} (confidence: {confidence.max():.2%})")