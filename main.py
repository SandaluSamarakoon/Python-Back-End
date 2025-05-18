from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import os
import uvicorn
from typing import Dict, Any
import logging
import soundfile as sf
import wave
import audioop
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Emotion Recognition API")

# Enable CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ideally, restrict this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
model_path = "D:/MIND BRIDGE/Python-Back-End/model.keras"  # Update this path to match your file structure
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # We'll handle the case where model isn't loaded in the prediction endpoint

def is_valid_wav(file_path):
    """Check if a WAV file is valid by trying to open it with wave"""
    try:
        with wave.open(file_path, 'rb') as wf:
            return True
    except Exception as e:
        logger.error(f"Invalid WAV file: {str(e)}")
        return False

def fix_wav_header(file_path):
    """Attempt to fix WAV file header issues by rewriting it"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # If it doesn't start with RIFF, it's not a WAV file or the header is corrupted
        if not data.startswith(b'RIFF'):
            logger.warning("WAV file doesn't start with RIFF header. Attempting to fix...")
            
            # Look for the audio data and create a new WAV file
            # This is a simple fix, might need adjustment based on actual file content
            with open(file_path + '.fixed.wav', 'wb') as f:
                # Create a basic WAV header
                channels = 1  # Assuming mono
                sample_width = 2  # Assuming 16-bit
                sample_rate = 44100  # Common sample rate
                
                # Find actual audio data (this is a simplified approach)
                # In a real file, you'd need to analyze the content more carefully
                audio_data = data
                
                # Create a new WAV file with proper header
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                
            return file_path + '.fixed.wav'
    except Exception as e:
        logger.error(f"Error fixing WAV header: {str(e)}")
    
    return file_path  # Return original path if fix failed

# Function to extract features using a more robust approach
def extract_features(file_path, n_mfcc=40):
    try:
        # Log the file info for debugging
        file_size = os.path.getsize(file_path)
        logger.info(f"Processing audio file: {file_path}, size: {file_size} bytes")
        
        # Verify WAV file and fix if necessary
        if not is_valid_wav(file_path):
            fixed_path = fix_wav_header(file_path)
            if fixed_path != file_path:
                logger.info(f"Using fixed WAV file: {fixed_path}")
                file_path = fixed_path
        
        # More robust audio loading with fallbacks
        try:
            # First try with soundfile (more reliable for WAV)
            audio, sample_rate = sf.read(file_path)
            logger.info(f"Loaded audio with soundfile, shape: {audio.shape}, sample_rate: {sample_rate}")
        except Exception as sf_error:
            logger.warning(f"Soundfile failed: {str(sf_error)}, trying librosa...")
            try:
                # Fall back to librosa with SR=None (use file's sample rate)
                audio, sample_rate = librosa.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
                logger.info(f"Loaded audio with librosa, length: {len(audio)}, sample_rate: {sample_rate}")
            except Exception as librosa_error:
                logger.error(f"Librosa also failed: {str(librosa_error)}")
                # As a last resort, try raw processing with wave
                try:
                    with wave.open(file_path, 'rb') as wf:
                        frames = wf.getnframes()
                        buffer = wf.readframes(frames)
                        sample_rate = wf.getframerate()
                        sample_width = wf.getsampwidth()
                        channels = wf.getnchannels()
                        
                        # Convert to mono if stereo
                        if channels == 2:
                            buffer = audioop.tomono(buffer, sample_width, 0.5, 0.5)
                        
                        # Convert buffer to numpy array
                        if sample_width == 2:  # 16-bit audio
                            audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sample_width == 4:  # 32-bit audio
                            audio = np.frombuffer(buffer, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:  # 8-bit audio
                            audio = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                        
                        logger.info(f"Loaded audio with wave, length: {len(audio)}, sample_rate: {sample_rate}")
                except Exception as wave_error:
                    logger.error(f"All audio loading methods failed: {str(wave_error)}")
                    raise Exception(f"Could not load audio with any method: {str(sf_error)} | {str(librosa_error)} | {str(wave_error)}")
        
        # Ensure audio is a numpy array and is mono
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # If audio is 2D (stereo), convert to mono
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        logger.info(f"Audio prepared: shape={audio.shape}, sample_rate={sample_rate}")
        
        # If audio is too short, pad it
        if len(audio) < sample_rate * 0.5:  # Less than 0.5 seconds
            logger.warning("Audio too short, padding")
            audio = np.pad(audio, (0, int(sample_rate * 0.5) - len(audio)), 'constant')
        
        # Extract MFCCs with error handling
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            logger.info(f"MFCCs extracted: shape={mfccs.shape}")
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {str(e)}")
            # Fallback: create dummy features if MFCC extraction fails
            logger.warning("Using fallback dummy features")
            mfccs = np.random.rand(n_mfcc, max(1, int(len(audio) / 512)))
        
        # Ensure shape is (40, 1) for model input
        mfccs = np.mean(mfccs, axis=1).reshape(n_mfcc, 1)
        return mfccs
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Voice Emotion Recognition API is running. Send a POST request to /predict with an audio file."}

# Add an error handler for the predict endpoint
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

@app.post("/predict")
async def predict_emotion(audio_file: UploadFile = File(...), expected_word: str = Form(None), category: str = Form(None)) -> Dict[str, Any]:
    global model
    
    logger.info(f"Received prediction request: file={audio_file.filename}, content_type={audio_file.content_type}")
    logger.info(f"Parameters: expected_word={expected_word}, category={category}")
    
    # Verify model is loaded
    if model is None:
        try:
            logger.info(f"Attempting to load model from {model_path}")
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.error(f"Model could not be loaded: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Model could not be loaded: {str(e)}"
            )
    
    # Accept more file types, not just .wav
    valid_audio_types = [".wav", ".webm", ".mp3", ".ogg", ".m4a"]
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    
    # For browsers that send .wav without extension
    if not file_ext and audio_file.content_type == 'audio/wav':
        file_ext = '.wav'
    
    if not file_ext and not any(audio_file.content_type.startswith(f"audio/{t.replace('.', '')}") for t in valid_audio_types):
        logger.warning(f"Unsupported content type: {audio_file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(valid_audio_types)}"
        )
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)
        
        logger.info(f"Saved uploaded file to {temp_file_path}, size: {len(content)} bytes")
        
        # Extract features with additional error handling
        try:
            features = extract_features(temp_file_path)
            features = np.expand_dims(features, axis=0)  # Add batch dimension (1, 40, 1)
            logger.info(f"Features extracted successfully, shape: {features.shape}")
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            # Return a friendly error instead of failing
            return {
                "filename": audio_file.filename,
                "predicted_emotion": "neutral",  # Default to neutral on failure
                "confidence_scores": {
                    "neutral": 1.0, "happy": 0.0, "sad": 0.0,
                    "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0
                },
                "text": "Could not process audio. Please try again and speak clearly.",
                "error": str(e),
                "matches": False
            }
        
        # Make prediction
        logger.info("Making prediction")
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction)
        
        # Define emotion labels (adjust based on your dataset)
        emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        
        # Convert numpy values to Python types for JSON serialization
        confidence_scores = prediction[0].tolist()
        
        logger.info(f"Prediction result: {emotions[predicted_label]}")
        
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
            # Also clean up the fixed file if it exists
            if os.path.exists(temp_file_path + '.fixed.wav'):
                os.unlink(temp_file_path + '.fixed.wav')
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")
        
        return {
            "filename": audio_file.filename,
            "predicted_emotion": emotions[predicted_label],
            "confidence_scores": {
                emotions[i]: float(confidence_scores[i]) for i in range(len(emotions))
            },
            "text": "placeholder for transcription",  # Your front-end expects this
            "matches": True if predicted_label == 0 else False  # Simple placeholder logic
        }
        
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                if os.path.exists(temp_file_path + '.fixed.wav'):
                    os.unlink(temp_file_path + '.fixed.wav')
            except:
                pass
        
        logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        # Return a user-friendly response instead of an error
        return {
            "filename": audio_file.filename,
            "predicted_emotion": "neutral",  # Default 
            "confidence_scores": {
                "neutral": 1.0, "happy": 0.0, "sad": 0.0,
                "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0
            },
            "text": "Error processing your speech. Please try again.",
            "error": str(e),
            "matches": False
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)