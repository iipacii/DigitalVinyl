from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import logging
from pathlib import Path
from .audio_encoder import encode_audio_to_image, decode_audio_from_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import predictions
from .genrePrediction.predict import predict
from .emotionPrediction.predict_emotion import predict_emotion

app = FastAPI(title="Audio Encoder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
UPLOAD_DIR = Path("app/static/uploads")
ENCODED_DIR = Path("app/static/encoded")
DECODED_DIR = Path("app/static/decoded")

for dir_path in [UPLOAD_DIR, ENCODED_DIR, DECODED_DIR]:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring directory exists: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {str(e)}")
        raise

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the home page"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving home page: {str(e)}")

@app.post("/encode")
async def encode_audio(
    audio_file: UploadFile = File(...),
    cover_image: UploadFile = File(...),
    song_name: str = Form(None),
):
    """Encode audio into an image with optional song name"""
    try:
        logger.info(f"Starting encoding process for audio file: {audio_file.filename}")
        logger.info(f"Original audio file details - Filename: {audio_file.filename}, Content-Type: {audio_file.content_type}")
        logger.info(f"Original cover image details - Filename: {cover_image.filename}, Content-Type: {cover_image.content_type}")
        
        # Validate audio file type
        audio_ext = os.path.splitext(audio_file.filename)[1].lower()
        logger.info(f"Audio file extension: {audio_ext}")
        
        if not audio_ext in ['.mp3', '.wav', '.ogg', '.m4a']:
            logger.error(f"Invalid audio format: {audio_ext}")
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {audio_ext}")
        
        # Validate image content type and set extension
        valid_image_types = {
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/png': '.png'
        }
        
        if cover_image.content_type not in valid_image_types:
            logger.error(f"Invalid image content type: {cover_image.content_type}")
            raise HTTPException(status_code=400, detail=f"Unsupported image type. Must be JPEG or PNG.")
        
        image_ext = valid_image_types[cover_image.content_type]
        logger.info(f"Using image extension based on content-type: {image_ext}")
        
        # Generate unique filenames - replace spaces with underscores and ensure proper extensions
        safe_audio_name = audio_file.filename.replace(' ', '_')
        safe_image_name = f"cover_image{image_ext}"  # Use a standard name with proper extension
        audio_filename = f"{uuid.uuid4()}-{safe_audio_name}"
        image_filename = f"{uuid.uuid4()}-{safe_image_name}"
        encoded_filename = f"encoded_{uuid.uuid4()}.png"
        
        logger.info(f"Generated safe filenames - Audio: {audio_filename}, Image: {image_filename}")
        
        # Save uploaded files
        audio_path = UPLOAD_DIR / audio_filename
        image_path = UPLOAD_DIR / image_filename
        encoded_path = ENCODED_DIR / encoded_filename
        
        logger.info(f"File paths - Audio: {audio_path}, Image: {image_path}, Encoded: {encoded_path}")
        
        # Write uploaded files to disk
        try:
            audio_content = await audio_file.read()
            if not audio_content:
                raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
                
            with open(audio_path, "wb") as audio_file_obj:
                audio_file_obj.write(audio_content)
                
            image_content = await cover_image.read()
            if not image_content:
                raise HTTPException(status_code=400, detail="Uploaded image file is empty")
                
            with open(image_path, "wb") as image_file_obj:
                image_file_obj.write(image_content)
        except Exception as e:
            logger.error(f"Error saving uploaded files: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving uploaded files: {str(e)}")
        
        # If song_name is empty, use filename without extension
        if not song_name:
            song_name = os.path.splitext(audio_file.filename)[0]
            logger.info(f"Using filename as song name: {song_name}")
        
        # Predict genre and emotion
        try:
            logger.info("Starting genre and emotion predictions")
            prediction_result = predict(str(audio_path))
            emotion_result = predict_emotion(str(audio_path))
            
            if "error" in prediction_result:
                logger.warning(f"Genre prediction warning: {prediction_result['error']}")
            if "error" in emotion_result:
                logger.warning(f"Emotion prediction warning: {emotion_result['error']}")
                
            logger.info(f"Genre prediction result: {prediction_result}")
            logger.info(f"Emotion prediction result: {emotion_result}")
        except Exception as e:
            logger.error(f"Error during predictions: {str(e)}")
            prediction_result = {"error": str(e)}
            emotion_result = {"error": str(e)}

        # Encode audio into image
        try:
            logger.info("Starting audio encoding process")
            # Get the predicted genre from the prediction result
            predicted_genre = prediction_result.get("genre", "Unknown") if "error" not in prediction_result else "Unknown"
            encode_audio_to_image(str(audio_path), str(image_path), str(encoded_path), song_name=song_name, genre=predicted_genre)
            logger.info(f"Audio encoding completed successfully, genre: {predicted_genre}")

            # Return the URL for the encoded image, genre prediction, and emotion prediction
            response = {"encoded_image_url": f"/static/encoded/{encoded_filename}"}
            if "error" not in prediction_result:
                response["genre"] = prediction_result["genre"]
                response["genre_confidence"] = prediction_result["confidence"]
            if "error" not in emotion_result:
                response["valence"] = emotion_result["valence"]
                response["arousal"] = emotion_result["arousal"]
                response["emotion"] = emotion_result["emotion"]
            
            return response
        except Exception as e:
            logger.error(f"Error during encoding: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during encoding: {str(e)}")
        finally:
            # Cleanup temporary files
            if audio_path.exists():
                audio_path.unlink()
            if image_path.exists():
                image_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during encoding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during encoding: {str(e)}")

@app.post("/decode")
async def decode_audio(
    encoded_image: UploadFile = File(...),
):
    """Decode audio from an encoded image"""
    try:
        logger.info(f"Starting decoding process for image: {encoded_image.filename}")
        
        # Validate file type
        image_ext = os.path.splitext(encoded_image.filename)[1].lower()
        if not image_ext in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail=f"Unsupported image format: {image_ext}")
        
        # Generate unique filenames
        image_filename = f"{uuid.uuid4()}{image_ext}"
        decoded_filename = f"decoded_{uuid.uuid4()}.tmp"
        
        # Save uploaded file
        image_path = UPLOAD_DIR / image_filename
        decoded_path = DECODED_DIR / decoded_filename
        
        logger.info(f"Saving encoded image to: {image_path}")
        
        # Write uploaded file to disk
        try:
            image_content = await encoded_image.read()
            if not image_content:
                raise HTTPException(status_code=400, detail="Uploaded image file is empty")
                
            with open(image_path, "wb") as image_file_obj:
                image_file_obj.write(image_content)
        except Exception as e:
            logger.error(f"Error saving uploaded image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving uploaded image: {str(e)}")
        
        # Decode audio from image and predict emotion
        try:
            logger.info("Starting audio decoding process")
            actual_decoded_path, song_name, genre = decode_audio_from_image(str(image_path), str(decoded_path))
            decoded_filename = os.path.basename(actual_decoded_path)
            logger.info(f"Audio successfully decoded to: {decoded_filename}")
            
            # Get emotion prediction for decoded audio
            try:
                emotion_result = predict_emotion(str(actual_decoded_path))
                if "error" in emotion_result:
                    logger.warning(f"Emotion prediction warning: {emotion_result['error']}")
                else:
                    logger.info(f"Emotion prediction result: {emotion_result}")
            except Exception as e:
                logger.error(f"Error during emotion prediction: {str(e)}")
                emotion_result = {"error": str(e)}
                
            if genre:
                logger.info(f"Decoded genre: {genre}")
        except Exception as e:
            logger.error(f"Error during decoding: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during decoding: {str(e)}")
        finally:
            # Cleanup temporary image file
            if image_path.exists():
                image_path.unlink()
        
        # Return the URL for the decoded audio, song name, genre and emotion if available
        result = {"decoded_audio_url": f"/static/decoded/{decoded_filename}"}
        if song_name:
            result["song_name"] = song_name
            logger.info(f"Decoded song name: {song_name}")
        if genre:
            result["genre"] = genre
            logger.info(f"Including genre in response: {genre}")
        if "error" not in emotion_result:
            result["valence"] = emotion_result["valence"]
            result["arousal"] = emotion_result["arousal"]
            result["emotion"] = emotion_result["emotion"]
            logger.info(f"Including emotion in response: {emotion_result['emotion']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during decoding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during decoding: {str(e)}")

@app.get("/encoded/{filename}")
async def get_encoded_image(filename: str):
    """Serve an encoded image"""
    try:
        file_path = ENCODED_DIR / filename
        if not file_path.exists():
            logger.error(f"Encoded image not found: {filename}")
            raise HTTPException(status_code=404, detail=f"Encoded image not found: {filename}")
        return FileResponse(file_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving encoded image {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving encoded image: {str(e)}")

@app.get("/decoded/{filename}")
async def get_decoded_audio(filename: str):
    """Serve a decoded audio file"""
    try:
        file_path = DECODED_DIR / filename
        if not file_path.exists():
            logger.error(f"Decoded audio not found: {filename}")
            raise HTTPException(status_code=404, detail=f"Decoded audio not found: {filename}")
        return FileResponse(file_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving decoded audio {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving decoded audio: {str(e)}")