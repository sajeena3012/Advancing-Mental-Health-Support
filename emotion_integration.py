import streamlit as st
import numpy as np
import os
import pickle
import random
import librosa
import torch
import speech_recognition as sr

from tensorflow.keras.models import load_model

# Emotion detection functions for audio and text

@st.cache_resource
def load_audio_emotion_model():
    """Load the audio emotion detection model"""
    try:
        audio_model = load_model("audio_emotion.h5")
        with open("label_encoder.pkl", "rb") as f:
            audio_label_encoder = pickle.load(f)
        return audio_model, audio_label_encoder, True
    except Exception as e:
        st.warning(f"Could not load audio emotion model: {e}")
        return None, None, False

# @st.cache_resource
# def load_text_emotion_model():
#     """Load the text emotion detection model"""
#     try:
#         text_model_dir = "trnsformer"
#         text_tokenizer = AutoTokenizer.from_pretrained(text_model_dir)
#         text_model = AutoModelForSequenceClassification.from_pretrained(text_model_dir)
#         text_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         text_model.to(text_device)
#         text_label_map = {0: "sad", 1: "happy", 2: "love", 3: "angry", 4: "fear", 5: "surprise"}
#         return text_model, text_tokenizer, text_device, text_label_map, True
#     except Exception as e:
#         st.warning(f"Could not load text emotion model: {e}")
#         return None, None, None, None, False

def noise_emotion(data):
    """Add random noise to the audio data for emotion detection"""
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch_emotion(data, rate=0.85):
    """Stretch the audio data for emotion detection"""
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch_emotion(data, sampling_rate, pitch_factor=0.7):
    """Change the pitch of the audio data for emotion detection"""
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features_emotion(data, sample_rate):
    """Extract audio features for emotion detection"""
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    return mfcc

def transform_audio_emotion(data, fns, sample_rate):
    """Transform audio data with random function for emotion detection"""
    fn = random.choice(fns)
    if fn == pitch_emotion:
        return fn(data, sample_rate)
    elif fn == "None":
        return data
    elif fn in [noise_emotion, stretch_emotion]:
        return fn(data)
    else:
        return data

def get_features_emotion(audio_path):
    """Get features from audio for emotion detection"""
    data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
    fns = [noise_emotion, pitch_emotion, "None"]
    features_list = []
    for _ in range(3):
        data_aug = transform_audio_emotion(data, fns, sample_rate)
        mfcc = extract_features_emotion(data_aug, sample_rate)
        mfcc = mfcc[:, :108]
        features_list.append(mfcc)
    return features_list

def predict_audio_emotion(audio_path):
    """Predict emotion from audio"""
    # Load the emotion model
    audio_model, audio_label_encoder, model_loaded = load_audio_emotion_model()
    if not model_loaded:
        return "Unknown"
    
    try:
        features = get_features_emotion(audio_path)
        predictions = []
        for feat in features:
            feat = np.expand_dims(feat, axis=0)
            feat = np.expand_dims(feat, axis=3)
            feat = np.swapaxes(feat, 1, 2)
            pred = audio_model.predict(feat)
            predictions.append(pred)
        avg_pred = np.mean(predictions, axis=0)
        emotion = audio_label_encoder.inverse_transform(avg_pred)[0]
        return emotion[0]
    except Exception as e:
        st.error(f"Error in audio emotion prediction: {e}")
        return "Unknown"
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Try to download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_text_emotion(input_data, input_type="text"):
    """Analyze emotion from text or audio input using VADER for text
    
    Args:
        input_data: The text string or audio file path
        input_type: "text" or "audio"
        
    Returns:
        String representing the detected emotion
    """
    if input_type == "audio":
        # For audio, you'd ideally transcribe it first
        # If you already have transcription in your main code, pass that text here
        # For now, let's return a default value
        return "Unknown"
    
    elif input_type == "text":
        # Return Unknown if text is empty or too short
        if not input_data or len(input_data.strip()) < 5:
            return "Unknown"
        
        # Initialize VADER
        sid = SentimentIntensityAnalyzer()
        
        # Get sentiment scores
        scores = sid.polarity_scores(input_data)
        
        # Extract emotional keywords for more specific emotions
        emotion_keywords = {
            'angry': ['angry', 'mad', 'furious', 'rage', 'anger', 'irritated', 'annoyed', 'frustrated'],
            'fear': ['afraid', 'scared', 'frightened', 'fearful', 'terrified', 'anxious', 'worry', 'nervous'],
            'surprise': ['surprised', 'shocked', 'astonished', 'amazed', 'unexpected', 'stunned'],
            'disgust': ['disgusted', 'gross', 'revolting', 'nauseating', 'repulsive'],
            'love': ['love', 'adore', 'cherish', 'affection', 'fond', 'loving']
        }
        
        # Check for specific emotions in text
        text_lower = input_data.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion.capitalize()
        
        # If no specific emotion keywords found, use VADER sentiment scores
        compound = scores['compound']
        
        # Map VADER scores to basic emotions
        if compound >= 0.5:
            return "Happy"
        elif compound <= -0.5:
            return "Sad"
        else:
            return "Neutral"

def transcribe_audio_emotion(audio_path):
    """Transcribe audio to text for emotion detection"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio_data)
        return transcription
    except Exception as e:
        st.warning(f"Transcription failed: {e}")
        return ""

# # This function can be called from the main application
def analyze_emotion(input_data, input_type="audio"):
    """
    Analyze emotion from either text or audio input
    
    Parameters:
    input_data (str or bytes): Either text content or path to audio file
    input_type (str): Either "text" or "audio"
    
    Returns:
    str: Detected emotion (e.g., "happy", "sad", "angry", etc.)
    """
    try:
        
            
        if input_type == "audio":
            # For audio input
            if not os.path.exists(input_data):
                return "Unknown"
                
            # First try to get emotion directly from audio
            audio_emotion = predict_audio_emotion(input_data)
            
           # Simple weighted voting (prioritize audio emotion)
            if audio_emotion != "Unknown":
                return audio_emotion
            
            else:
                return "Unknown"
            
            
    except Exception as e:
        st.error(f"Error analyzing emotion: {e}")
        return "Unknown"