#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Genre Prediction Script

This script loads a pre-trained model and predicts the genre of an audio file.
It outputs the result in JSON format.

Usage:
    python predict.py /path/to/audio_file.mp3

Output format:
    {"genre": "predicted_genre", "confidence": 0.95}
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

# --- Constants ---
SR = 22050         # Sample Rate
N_MFCC = 40        # Number of MFCCs to extract
N_FFT = 2048       # Window size for FFT
HOP_LENGTH = 512   # Hop length for STFT
FIXED_LENGTH = 128 # Number of time frames for MFCC features

# Path to the trained model
MODEL_PATH = 'best_genre_classifier.pth'

# --- Model Definition ---
class GenreClassifier(nn.Module):
    """Simple CNN model for Genre Classification based on MFCCs."""
    def __init__(self, input_shape, num_classes):
        # input_shape expected as (channels, n_mfcc, fixed_length) e.g., (1, 40, 128)
        super(GenreClassifier, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.3)

        # Calculate the size of the flattened features dynamically
        self._to_linear = self._get_conv_output_size(input_shape)

        # Fully Connected Layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.bn4 = nn.BatchNorm1d(512) # BatchNorm for FC layer
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes) # Output layer

    def _get_conv_output_size(self, shape):
        """Helper to calculate the flattened size after conv layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape) # Batch size 1
            output = self._forward_features(dummy_input)
            return int(np.prod(output.size()))

    def _forward_features(self, x):
        """Forward pass through convolutional layers."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        return x

    def forward(self, x):
        """Full forward pass."""
        x = self._forward_features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x) # Raw logits output
        return x

# --- Feature Extraction Function ---
def extract_features(file_path, n_mfcc=N_MFCC, target_sr=SR, n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, fixed_length=FIXED_LENGTH):
    """
    Extract MFCCs using torchaudio with robust handling and normalization.
    Pads or truncates to a fixed length.
    """
    try:
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(file_path)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr # Update sample_rate after resampling

        # Create MFCC transform
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': 128, # Typically more mels than MFCCs
                'center': True,
            }
        )

        # Extract MFCCs
        mfccs = mfcc_transform(waveform) # Output shape: (channel, n_mfcc, time)

        # Remove channel dimension and convert to numpy
        mfccs = mfccs.squeeze(0).numpy() # Shape: (n_mfcc, time)

        # Handle length consistency
        current_length = mfccs.shape[1]
        if fixed_length is not None: # Only pad/truncate if fixed_length is specified
            if current_length < fixed_length:
                pad_width = fixed_length - current_length
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            elif current_length > fixed_length:
                mfccs = mfccs[:, :fixed_length]

        # Normalize (mean/std deviation)
        mean = np.mean(mfccs)
        std = np.std(mfccs)
        mfccs = (mfccs - mean) / (std + 1e-8) # Add epsilon for stability

        return mfccs

    except Exception as e:
        print(f"Error extracting features: {e}", file=sys.stderr)
        return None

# --- Prediction Function ---
def predict(file_path, model_path=MODEL_PATH):
    """
    Predict the genre of an audio file.
    Returns a dictionary with genre and confidence.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Extract features
        features = extract_features(file_path)
        if features is None:
            return {"error": "Failed to extract audio features"}
        
        # Define the genre classes - these must match what the model was trained on
        # These are the 8 top-level genres in the FMA dataset
        genres = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 
                 'Instrumental', 'International', 'Pop', 'Rock']
        
        # Create model
        input_shape = (1, N_MFCC, FIXED_LENGTH)
        model = GenreClassifier(input_shape=input_shape, num_classes=len(genres))
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Prepare input tensor
        # Add batch and channel dimensions
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        # Get predicted genre and confidence
        predicted_genre = genres[predicted_idx.item()]
        confidence_score = confidence.item()
        
        return {
            "genre": predicted_genre,
            "confidence": confidence_score
        }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# --- Main Function ---
def main():
    # Check for command line arguments
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <audio_file_path>", file=sys.stderr)
        sys.exit(1)
    
    # Get the file path from command line
    file_path = sys.argv[1]
    
    # Get prediction
    result = predict(file_path)
    
    # Output as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()