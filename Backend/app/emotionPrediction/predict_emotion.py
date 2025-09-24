#!/usr/bin/env python3
"""
Audio Emotion Prediction Script

This script loads a pre-trained emotion model and makes predictions on audio files,
outputting valence and arousal values in JSON format.

Usage:
    python predict_emotion.py path_to_audio_file.mp3

Output:
    JSON with valence and arousal values
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import librosa
import argparse

# Configuration
SAMPLE_RATE = 22050
SEGMENT_DURATION = 30  # seconds
N_MELS = 128
SPEC_WIDTH = 128
MODEL_PATH = os.path.join(os.path.dirname(__file__),'deam_emotion_model.pth')

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming input size is 128x128 after pooling
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # Output: valence and arousal
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For 0-1 output range
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, 128 * 8 * 8)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))  # Use sigmoid to get values in [0,1]
        
        return x

def predict_emotion(audio_file, model_path=MODEL_PATH):
    """
    Predict emotion (valence, arousal) for an audio file
    
    Args:
        audio_file: Path to the audio file
        model_path: Path to the saved model
        
    Returns:
        Dictionary with valence and arousal values
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Load model
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load audio file
    try:
        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=SEGMENT_DURATION)
    except Exception as e:
        raise Exception(f"Error loading audio file: {str(e)}")
    
    # If audio is shorter than segment_duration, pad with zeros
    if len(y) < SEGMENT_DURATION * SAMPLE_RATE:
        y = np.pad(y, (0, SEGMENT_DURATION * SAMPLE_RATE - len(y)), 'constant')
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    
    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Resize to target width
    if mel_spec_norm.shape[1] > SPEC_WIDTH:
        mel_spec_norm = mel_spec_norm[:, :SPEC_WIDTH]
    elif mel_spec_norm.shape[1] < SPEC_WIDTH:
        padding = np.zeros((N_MELS, SPEC_WIDTH - mel_spec_norm.shape[1]))
        mel_spec_norm = np.hstack((mel_spec_norm, padding))
    
    # Convert to tensor
    mel_spec_tensor = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    mel_spec_tensor = mel_spec_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(mel_spec_tensor)
    
    valence = float(prediction[0][0].item())
    arousal = float(prediction[0][1].item())
    
    # Map to emotion categories (optional)
    emotion_map = {
        (True, True): "Happy",     # High valence, high arousal
        (True, False): "Calm",     # High valence, low arousal
        (False, True): "Angry",    # Low valence, high arousal
        (False, False): "Sad"      # Low valence, low arousal
    }
    
    emotion = emotion_map[(valence > 0.5, arousal > 0.5)]
    
    return {
        "valence": valence,
        "arousal": arousal,
        "emotion": emotion
    }

def main():
    parser = argparse.ArgumentParser(description='Predict emotion from audio file')
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to the model file')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print the JSON output')
    
    args = parser.parse_args()
    
    try:
        result = predict_emotion(args.audio_file, args.model)
        
        # Output as JSON
        if args.pretty:
            print(json.dumps(result, indent=4))
        else:
            print(json.dumps(result))
            
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())