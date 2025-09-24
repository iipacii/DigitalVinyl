# Audio Encoder - Virtual Vinyl

A web application that encodes audio files into images, creating "virtual vinyl" records that can be shared and decoded.

## Features

- Encode audio files into images (creating virtual vinyl)
- Decode audio from encoded images
- Automatic audio downsampling to fit within image capacity
- Automatic image processing to ensure square format
- Simple, user-friendly web interface

## Technical Details

This application uses steganography techniques to embed audio data within the least significant bits of image pixels. The process allows for hiding audio files within images with minimal visual impact, creating shareable "virtual vinyl" records.

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- NumPy
- Pillow (PIL)
- SciPy
- python-multipart

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install fastapi uvicorn numpy pillow scipy python-multipart
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```
2. Open your web browser and navigate to `http://localhost:8000`
3. Use the interface to encode and decode audio files

## How It Works

1. **Audio Encoding**:
   - The audio file is downsampled to reduce size while maintaining quality
   - The cover image is processed to ensure it's square
   - Audio data is embedded in the least significant bits of the image pixels
   - A terminator sequence is added to mark the end of the audio data

2. **Audio Decoding**:
   - The encoded image is analyzed to extract the least significant bits
   - The sample rate is retrieved from the header
   - Audio data is reconstructed and saved as a WAV file

## Limitations

- Image size limits the amount of audio that can be encoded
- Higher quality audio requires larger images
- The encoding process reduces audio quality to fit the image capacity

## License

MIT