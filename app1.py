# # # import streamlit as st
# # # import json
# # # import os
# # # import re
# # # import string
# # # from st_audiorec import st_audiorec
# # # import io
# # # import whisper
# # # from nltk.corpus import stopwords
# # # from nltk.tokenize import word_tokenize
# # # from nltk.stem import WordNetLemmatizer
# # # import speech_recognition as sr
# # # from pydub import AudioSegment
# # # import pickle
# # # import numpy as np
# # # import librosa
# # # import tensorflow as tf
# # # from tensorflow.keras.models import load_model
# # # from sklearn.feature_extraction.text import TfidfVectorizer

# # # # Initialize session state
# # # session_state = st.session_state
# # # if "user_index" not in st.session_state:
# # #     st.session_state["user_index"] = 0

# # # # Load text classification model
# # # with open('logistic.pkl', 'rb') as f:
# # #     text_model = pickle.load(f)

# # # with open('tfidf_vectorizer.pik', 'rb') as f:
# # #     vectorizer = pickle.load(f)
    
# # # # Load voice depression detection model
# # # try:
# # #     voice_model = load_model("mlp_model.keras")
# # # except:
# # #     # If model doesn't exist, we'll need to inform the user
# # #     voice_model = None

# # # # Helper functions for voice feature extraction
# # # def noise(data):
# # #     """Add random noise to the audio data"""
# # #     noise_amp = 0.035*np.random.uniform()*np.amax(data)
# # #     data = data + noise_amp*np.random.normal(size=data.shape[0])
# # #     return data

# # # def stretch(data, rate=0.8):
# # #     """Stretch the audio data"""
# # #     return librosa.effects.time_stretch(data, rate=rate)

# # # def pitch(data, sampling_rate, pitch_factor=0.7):
# # #     """Change the pitch of the audio data"""
# # #     return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

# # # def extract_features(data, sample_rate):
# # #     """Extract audio features for depression detection"""
# # #     # ZCR
# # #     result = np.array([])
# # #     zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
# # #     result = np.hstack((result, zcr))  # stacking horizontally

# # #     # Chroma_stft
# # #     stft = np.abs(librosa.stft(data))
# # #     chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
# # #     result = np.hstack((result, chroma_stft))  # stacking horizontally

# # #     # MFCC
# # #     mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
# # #     result = np.hstack((result, mfcc))  # stacking horizontally

# # #     # Root Mean Square Value
# # #     rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
# # #     result = np.hstack((result, rms))  # stacking horizontally

# # #     # MelSpectogram
# # #     mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
# # #     result = np.hstack((result, mel))  # stacking horizontally

# # #     # Spectral Contrast
# # #     spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
# # #     result = np.hstack((result, spectral_contrast))

# # #     # Tonnetz
# # #     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
# # #     result = np.hstack((result, tonnetz))

# # #     # Rolloff
# # #     rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
# # #     result = np.hstack((result, rolloff))

# # #     # Zero Crossing Rate Variance
# # #     zcr_var = np.var(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
# # #     result = np.hstack((result, zcr_var))

# # #     return result

# # # def analyze_voice(audio_path, sample_rate=22050):
# # #     """Analyze voice for depression detection"""
# # #     if voice_model is None:
# # #         return "Voice model not loaded", 0.0
    
# # #     try:
# # #         # Extract features from the audio
# # #         data, _ = librosa.load(audio_path, sr=sample_rate, duration=2.5, offset=0.6)
# # #         features = extract_features(data, sample_rate)
        
# # #         # Reshape features for model input
# # #         features = np.array([features])
        
# # #         # Make prediction
# # #         prediction = voice_model.predict(features)
# # #         depression_prob = prediction[0][1]  # Assuming index 1 is depression
        
# # #         # Determine result based on probability
# # #         if depression_prob > 0.5:
# # #             return "depression", depression_prob
# # #         else:
# # #             return "normal", depression_prob
# # #     except Exception as e:
# # #         st.error(f"Error analyzing voice: {e}")
# # #         return "error", 0.0

# # # def signup(json_file_path="data.json"):
# # #     st.title("Signup Page")
# # #     with st.form("signup_form"):
# # #         st.write("Fill in the details below to create an account:")
# # #         name = st.text_input("Name:")
# # #         email = st.text_input("Email:")
# # #         age = st.number_input("Age:", min_value=0, max_value=120)
# # #         sex = st.radio("Sex:", ("Male", "Female", "Other"))
# # #         password = st.text_input("Password:", type="password")
# # #         confirm_password = st.text_input("Confirm Password:", type="password")

# # #         if st.form_submit_button("Signup"):
# # #             if password == confirm_password:
# # #                 user = create_account(name, email, age, sex, password, json_file_path)
# # #                 session_state["logged_in"] = True
# # #                 session_state["user_info"] = user
# # #             else:
# # #                 st.error("Passwords do not match. Please try again.")

# # # def check_login(username, password, json_file_path="data.json"):
# # #     try:
# # #         with open(json_file_path, "r") as json_file:
# # #             data = json.load(json_file)

# # #         for user in data["users"]:
# # #             if user["email"] == username and user["password"] == password:
# # #                 session_state["logged_in"] = True
# # #                 session_state["user_info"] = user
# # #                 st.success("Login successful!")
# # #                 return user

# # #         st.error("Invalid credentials. Please try again.")
# # #         return None
# # #     except Exception as e:
# # #         st.error(f"Error checking login: {e}")
# # #         return None

# # # def initialize_database(json_file_path="data.json"):
# # #     try:
# # #         # Check if JSON file exists
# # #         if not os.path.exists(json_file_path):
# # #             # Create an empty JSON structure
# # #             data = {"users": []}
# # #             with open(json_file_path, "w") as json_file:
# # #                 json.dump(data, json_file)
# # #     except Exception as e:
# # #         print(f"Error initializing database: {e}")
        
# # # def create_account(name, email, age, sex, password, json_file_path="data.json"):
# # #     try:
# # #         # Check if the JSON file exists or is empty
# # #         if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
# # #             data = {"users": []}
# # #         else:
# # #             with open(json_file_path, "r") as json_file:
# # #                 data = json.load(json_file)

# # #         # Append new user data to the JSON structure
# # #         user_info = {
# # #             "name": name,
# # #             "email": email,
# # #             "age": age,
# # #             "sex": sex,
# # #             "password": password,
# # #         }
# # #         data["users"].append(user_info)

# # #         # Save the updated data to JSON
# # #         with open(json_file_path, "w") as json_file:
# # #             json.dump(data, json_file, indent=4)

# # #         st.success("Account created successfully! You can now login.")
# # #         return user_info
# # #     except json.JSONDecodeError as e:
# # #         st.error(f"Error decoding JSON: {e}")
# # #         return None
# # #     except Exception as e:
# # #         st.error(f"Error creating account: {e}")
# # #         return None

# # # def login(json_file_path="data.json"):
# # #     st.title("Login Page")
# # #     username = st.text_input("Email:")
# # #     password = st.text_input("Password:", type="password")

# # #     login_button = st.button("Login")

# # #     if login_button:
# # #         user = check_login(username, password, json_file_path)
# # #         if user is not None:
# # #             session_state["logged_in"] = True
# # #             session_state["user_info"] = user
# # #         else:
# # #             st.error("Invalid credentials. Please try again.")

# # # def get_user_info(email, json_file_path="data.json"):
# # #     try:
# # #         with open(json_file_path, "r") as json_file:
# # #             data = json.load(json_file)
# # #             for user in data["users"]:
# # #                 if user["email"] == email:
# # #                     return user
# # #         return None
# # #     except Exception as e:
# # #         st.error(f"Error getting user information: {e}")
# # #         return None

# # # def render_dashboard(user_info, json_file_path="data.json"):
# # #     try:
# # #         st.title(f"Welcome to the Dashboard, {user_info['name']}!")
# # #         st.subheader("User Information:")
# # #         st.write(f"Name: {user_info['name']}")
# # #         st.write(f"Sex: {user_info['sex']}")
# # #         st.write(f"Age: {user_info['age']}")

# # #     except Exception as e:
# # #         st.error(f"Error rendering dashboard: {e}")
    
# # # def preprocess(text): 
# # #     def remove_url(text):
# # #         return re.sub(r"http\S+", "", text) 
# # #     exclude = string.punctuation
# # #     def remove_punctuation(text):
# # #         return text.translate(str.maketrans("", "", exclude))
    
# # #     def remove_stopwords(text):
# # #         stopword = stopwords.words('english')
# # #         new_text = []
# # #         for word in text.split():
# # #             if word in stopword:
# # #                 new_text.append('')
# # #             else:
# # #                 new_text.append(word)
# # #         x = new_text[:]
# # #         new_text.clear()
# # #         return " ".join(x)
# # #     def lemmatize_text(text):
# # #         words = word_tokenize(text)
# # #         lemmatizer = WordNetLemmatizer()
# # #         lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
# # #         return " ".join(lemmatized_words)
# # #     text = remove_url(text)
# # #     text = remove_punctuation(text)
# # #     text = remove_stopwords(text)
# # #     text = lemmatize_text(text)
# # #     return text
   
# # # def transcribe_audio_from_data(file_data):
# # #     with open("temp.mp3", "wb") as f:
# # #         f.write(file_data)
# # #     model = whisper.load_model("base")
# # #     result = model.transcribe("temp.mp3")
# # #     os.remove("temp.mp3")
# # #     return result['text']

# # # def get_combined_diagnosis(voice_result, text_result):
# # #     """Combine voice and text analysis results to get a final diagnosis"""
# # #     # Define rules for combined diagnosis
# # #     if voice_result == "depression" and text_result == "suicide":
# # #         return "High risk - Suicidal tendencies detected", "danger"
# # #     elif voice_result == "depression" and text_result == "depression":
# # #         return "Moderate risk - Clinical depression likely", "warning"
# # #     elif voice_result == "depression" and text_result == "normal":
# # #         return "Low risk - Mild depression possible", "info"
# # #     elif voice_result == "normal" and text_result == "depression":
# # #         return "Low risk - Situational depression possible", "info"
# # #     elif voice_result == "normal" and text_result == "suicide":
# # #         return "Moderate risk - Suicidal ideation detected", "warning"
# # #     elif voice_result == "normal" and text_result == "normal":
# # #         return "No risk - Normal mental state", "success"
# # #     else:
# # #         return "Assessment inconclusive", "info"

# # # def main(json_file_path="data.json"):
# # #     st.sidebar.title("Mental Health Assessment")
# # #     page = st.sidebar.radio(
# # #         "Go to",
# # #         ("Signup/Login", "Dashboard", "Mental Health Assessment"),
# # #         key="mental_health_assessment",
# # #     )

# # #     if page == "Signup/Login":
# # #         st.title("Signup/Login Page")
# # #         login_or_signup = st.radio(
# # #             "Select an option", ("Login", "Signup"), key="login_signup"
# # #         )
# # #         if login_or_signup == "Login":
# # #             login(json_file_path)
# # #         else:
# # #             signup(json_file_path)

# # #     elif page == "Dashboard":
# # #         if session_state.get("logged_in"):
# # #             render_dashboard(session_state["user_info"])
# # #         else:
# # #             st.warning("Please login/signup to view the dashboard.")

# # #     elif page == "Mental Health Assessment":
# # #         if session_state.get("logged_in"):
# # #             st.title("Mental Health Assessment")
            
# # #             # Initialize session state variables if they don't exist
# # #             if "voice_result" not in st.session_state:
# # #                 st.session_state.voice_result = None
# # #                 st.session_state.voice_prob = 0.0
# # #                 st.session_state.text_result = None
# # #                 st.session_state.audio_data = None
# # #                 st.session_state.analysis_complete = False
            
# # #             # Voice Analysis Section
# # #             st.header("Voice Analysis")
# # #             options = ["Record", "Upload"]
# # #             choice = st.radio("Choose an option for voice input", options)
            
# # #             if choice == "Record":
# # #                 st.write("Click the button below to start recording:")
# # #                 audio = st_audiorec()
# # #                 if audio is not None:
# # #                     st.session_state.audio_data = audio
# # #                     if st.button('Analyze Recording'):
# # #                         # Save audio for analysis
# # #                         with open("temp_analysis.wav", "wb") as f:
# # #                             f.write(audio)
                        
# # #                         # Analyze voice for depression
# # #                         st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                        
# # #                         # Analyze with text model - Done separately to avoid showing transcription
# # #                         try:
# # #                             # Don't display the transcribed text
# # #                             transcribed_text = transcribe_audio_from_data(audio)
                            
# # #                             # Process text using the TF-IDF vectorizer that the model was trained with
# # #                             processed_text = preprocess(transcribed_text)
                            
# # #                             # Use the vectorizer to transform the text
# # #                             tfidf_vector = vectorizer.transform([processed_text])
                            
# # #                             # Check if features match what the model expects
# # #                             text_pred = text_model.predict(tfidf_vector)
# # #                             st.session_state.text_result = text_pred[0]
                            
# # #                         except ValueError as e:
# # #                             # If there's a feature mismatch or other error, default to using only voice analysis
# # #                             st.error("Error in text analysis. Using only voice analysis for diagnosis.")
# # #                             st.session_state.text_result = st.session_state.voice_result
                        
# # #                         st.session_state.analysis_complete = True
                        
# # #                         # Remove temporary file
# # #                         if os.path.exists("temp_analysis.wav"):
# # #                             os.remove("temp_analysis.wav")
            
# # #             elif choice == "Upload":
# # #                 audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
# # #                 if audio is not None:
# # #                     st.audio(audio, format="audio/wav")
# # #                     st.session_state.audio_data = audio.read()
                    
# # #                     if st.button('Analyze Upload'):
# # #                         # Save audio for analysis
# # #                         with open("temp_analysis.wav", "wb") as f:
# # #                             f.write(st.session_state.audio_data)
                        
# # #                         # Analyze voice for depression
# # #                         st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                        
# # #                         # Analyze with text model - Done separately to avoid showing transcription
# # #                         try:
# # #                             # Don't display the transcribed text
# # #                             transcribed_text = transcribe_audio_from_data(st.session_state.audio_data)
                            
# # #                             # Process text using the TF-IDF vectorizer that the model was trained with
# # #                             processed_text = preprocess(transcribed_text)
                            
# # #                             # Use the vectorizer to transform the text
# # #                             tfidf_vector = vectorizer.transform([processed_text])
                            
# # #                             # Check if features match what the model expects
# # #                             text_pred = text_model.predict(tfidf_vector)
# # #                             st.session_state.text_result = text_pred[0]
                            
# # #                         except ValueError as e:
# # #                             # If there's a feature mismatch or other error, default to using only voice analysis
# # #                             st.error("Error in text analysis. Using only voice analysis for diagnosis.")
# # #                             st.session_state.text_result = st.session_state.voice_result
                        
# # #                         st.session_state.analysis_complete = True
                        
# # #                         # Remove temporary file
# # #                         if os.path.exists("temp_analysis.wav"):
# # #                             os.remove("temp_analysis.wav")
            
# # #             # Display results
# # #             if st.session_state.analysis_complete:
# # #                 st.header("Assessment Results")
                
# # #                 # Display voice analysis results
# # #                 st.subheader("Voice Analysis:")
# # #                 if st.session_state.voice_result:
# # #                     if st.session_state.voice_result == "depression":
# # #                         st.warning(f"Voice indicates depression tendencies (Confidence: {st.session_state.voice_prob:.2f})")
# # #                     elif st.session_state.voice_result == "normal":
# # #                         st.success(f"Voice indicates normal mental state (Confidence: {1-st.session_state.voice_prob:.2f})")
# # #                     else:
# # #                         st.info("Voice analysis inconclusive")
                
# # #                 # Display combined diagnosis
# # #                 st.subheader("Final Assessment:")
# # #                 if st.session_state.voice_result and st.session_state.text_result:
# # #                     diagnosis, status = get_combined_diagnosis(
# # #                         st.session_state.voice_result, 
# # #                         st.session_state.text_result
# # #                     )
                    
# # #                     if status == "danger":
# # #                         st.error(diagnosis)
# # #                     elif status == "warning":
# # #                         st.warning(diagnosis)
# # #                     elif status == "success":
# # #                         st.success(diagnosis)
# # #                     else:
# # #                         st.info(diagnosis)
                    
# # #                     # Provide resources based on diagnosis
# # #                     st.subheader("Resources:")
# # #                     if "Suicidal" in diagnosis:
# # #                         st.write("Some resources for controlling suicidal tendencies:")
# # #                         st.markdown("- [Are you feeling Suicidal?](https://www.helpguide.org/articles/suicide-prevention/are-you-feeling-suicidal.htm)")
# # #                         st.markdown("- [Suicide and Suicidal Thoughts](https://www.mayoclinic.org/diseases-conditions/suicide/symptoms-causes/syc-20378048)")
# # #                         st.markdown("- [Suicide Prevention](https://www.ted.com/talks/ashleigh_husbands_suicide_prevention?hasSummary=true)")
# # #                         st.markdown("- [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/)")
# # #                         st.markdown("- [Crisis Text Line](https://www.crisistextline.org/)")
# # #                         st.markdown("- [International Association for Suicide Prevention](https://www.iasp.info/resources/Crisis_Centres/)")
# # #                     elif "depression" in diagnosis.lower():
# # #                         st.write("Here are some resources to help cope with depression:")
# # #                         st.markdown("- [How to cope up with depression](https://www.youtube.com/watch?v=8Su5VtKeXU8)")
# # #                         st.markdown("- [Ways to cope with depression](https://www.medicalnewstoday.com/articles/327018)")
# # #                         st.markdown("- [Depression Treatment](https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/depression-treatment-and-management)")
# # #                         st.markdown("- [Don't suffer from Depression in Silence](https://www.ted.com/talks/nikki_webber_allen_don_t_suffer_from_your_depression_in_silence?hasSummary=true&language=en)")
# # #                     elif "Normal" in diagnosis:
# # #                         st.write("Great job! Keep up the positivity and stay clear of negative thinking.")
# # #         else:
# # #             st.warning("Please login/signup to use the app.")
            
# # # if __name__ == "__main__":
# # #     initialize_database()
# # #     main()

# # import streamlit as st
# # import json
# # import os
# # import re
# # import string
# # from st_audiorec import st_audiorec
# # import io
# # import whisper
# # from nltk.corpus import stopwords
# # from nltk.tokenize import word_tokenize
# # from nltk.stem import WordNetLemmatizer
# # import speech_recognition as sr
# # from pydub import AudioSegment
# # import pickle
# # import numpy as np
# # import librosa
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # from sklearn.feature_extraction.text import TfidfVectorizer

# # # Initialize session state
# # session_state = st.session_state
# # if "user_index" not in st.session_state:
# #     st.session_state["user_index"] = 0

# # # Load text classification model
# # try:
# #     with open('logistic.pkl', 'rb') as f:
# #         text_model = pickle.load(f)

# #     with open('tfidf_vectorizer.pik', 'rb') as f:
# #         vectorizer = pickle.load(f)
# #     text_model_loaded = True
# # except Exception as e:
# #     st.error(f"Error loading text model: {e}")
# #     text_model_loaded = False
    
# # # Load voice depression detection model
# # try:
# #     voice_model = load_model(r"c:\Users\praga\Documents\Projects\Depresion-Sucide\file (14)\kaggle\working\rf_model.pkl")
# #     voice_model_loaded = True
# # except Exception as e:
# #     st.error(f"Error loading voice model: {e}")
# #     voice_model_loaded = False

# # # Helper functions for voice feature extraction
# # def noise(data):
# #     """Add random noise to the audio data"""
# #     noise_amp = 0.035*np.random.uniform()*np.amax(data)
# #     data = data + noise_amp*np.random.normal(size=data.shape[0])
# #     return data

# # def stretch(data, rate=0.8):
# #     """Stretch the audio data"""
# #     return librosa.effects.time_stretch(data, rate=rate)

# # def pitch(data, sampling_rate, pitch_factor=0.7):
# #     """Change the pitch of the audio data"""
# #     return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

# # def extract_features(data, sample_rate):
# #     """Extract audio features for depression detection"""
# #     # ZCR
# #     result = np.array([])
# #     zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
# #     result = np.hstack((result, zcr))  # stacking horizontally

# #     # Chroma_stft
# #     stft = np.abs(librosa.stft(data))
# #     chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, chroma_stft))  # stacking horizontally

# #     # MFCC
# #     mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, mfcc))  # stacking horizontally

# #     # Root Mean Square Value
# #     rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
# #     result = np.hstack((result, rms))  # stacking horizontally

# #     # MelSpectogram
# #     mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, mel))  # stacking horizontally

# #     # Spectral Contrast
# #     spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, spectral_contrast))

# #     # Tonnetz
# #     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, tonnetz))

# #     # Rolloff
# #     rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, rolloff))

# #     # Zero Crossing Rate Variance
# #     zcr_var = np.var(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
# #     result = np.hstack((result, zcr_var))

# #     return result

# # def analyze_voice(audio_path, sample_rate=22050):
# #     """Analyze voice for depression detection"""
# #     if not voice_model_loaded:
# #         return "model_not_loaded", 0.0
    
# #     try:
# #         # Extract features from the audio
# #         data, _ = librosa.load(audio_path, sr=sample_rate, duration=2.5, offset=0.6)
# #         features = extract_features(data, sample_rate)
        
# #         # Reshape features for model input
# #         features = np.array([features])
        
# #         # Make prediction
# #         prediction = voice_model.predict(features)
# #         st.write(prediction)
# #         y_pred = np.argmax(prediction, axis=1)
# #         st.write(y_pred)
# #         depression_prob = prediction[0][1]  # Assuming index 1 is depression
        
# #         # Determine result based on probability
# #         if depression_prob > 0.5:
# #             return "depression", depression_prob
# #         else:
# #             return "normal", depression_prob
# #     except Exception as e:
# #         st.error(f"Error analyzing voice: {e}")
# #         return "error", 0.0

# # def signup(json_file_path="data.json"):
# #     st.title("Signup Page")
# #     with st.form("signup_form"):
# #         st.write("Fill in the details below to create an account:")
# #         name = st.text_input("Name:")
# #         email = st.text_input("Email:")
# #         age = st.number_input("Age:", min_value=0, max_value=120)
# #         sex = st.radio("Sex:", ("Male", "Female", "Other"))
# #         password = st.text_input("Password:", type="password")
# #         confirm_password = st.text_input("Confirm Password:", type="password")

# #         if st.form_submit_button("Signup"):
# #             if password == confirm_password:
# #                 user = create_account(name, email, age, sex, password, json_file_path)
# #                 session_state["logged_in"] = True
# #                 session_state["user_info"] = user
# #             else:
# #                 st.error("Passwords do not match. Please try again.")

# # def check_login(username, password, json_file_path="data.json"):
# #     try:
# #         with open(json_file_path, "r") as json_file:
# #             data = json.load(json_file)

# #         for user in data["users"]:
# #             if user["email"] == username and user["password"] == password:
# #                 session_state["logged_in"] = True
# #                 session_state["user_info"] = user
# #                 st.success("Login successful!")
# #                 return user

# #         st.error("Invalid credentials. Please try again.")
# #         return None
# #     except Exception as e:
# #         st.error(f"Error checking login: {e}")
# #         return None

# # def initialize_database(json_file_path="data.json"):
# #     try:
# #         # Check if JSON file exists
# #         if not os.path.exists(json_file_path):
# #             # Create an empty JSON structure
# #             data = {"users": []}
# #             with open(json_file_path, "w") as json_file:
# #                 json.dump(data, json_file)
# #     except Exception as e:
# #         print(f"Error initializing database: {e}")
        
# # def create_account(name, email, age, sex, password, json_file_path="data.json"):
# #     try:
# #         # Check if the JSON file exists or is empty
# #         if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
# #             data = {"users": []}
# #         else:
# #             with open(json_file_path, "r") as json_file:
# #                 data = json.load(json_file)

# #         # Append new user data to the JSON structure
# #         user_info = {
# #             "name": name,
# #             "email": email,
# #             "age": age,
# #             "sex": sex,
# #             "password": password,
# #         }
# #         data["users"].append(user_info)

# #         # Save the updated data to JSON
# #         with open(json_file_path, "w") as json_file:
# #             json.dump(data, json_file, indent=4)

# #         st.success("Account created successfully! You can now login.")
# #         return user_info
# #     except json.JSONDecodeError as e:
# #         st.error(f"Error decoding JSON: {e}")
# #         return None
# #     except Exception as e:
# #         st.error(f"Error creating account: {e}")
# #         return None

# # def login(json_file_path="data.json"):
# #     st.title("Login Page")
# #     username = st.text_input("Email:")
# #     password = st.text_input("Password:", type="password")

# #     login_button = st.button("Login")

# #     if login_button:
# #         user = check_login(username, password, json_file_path)
# #         if user is not None:
# #             session_state["logged_in"] = True
# #             session_state["user_info"] = user
# #         else:
# #             st.error("Invalid credentials. Please try again.")

# # def get_user_info(email, json_file_path="data.json"):
# #     try:
# #         with open(json_file_path, "r") as json_file:
# #             data = json.load(json_file)
# #             for user in data["users"]:
# #                 if user["email"] == email:
# #                     return user
# #         return None
# #     except Exception as e:
# #         st.error(f"Error getting user information: {e}")
# #         return None

# # def render_dashboard(user_info, json_file_path="data.json"):
# #     try:
# #         st.title(f"Welcome to the Dashboard, {user_info['name']}!")
# #         st.subheader("User Information:")
# #         st.write(f"Name: {user_info['name']}")
# #         st.write(f"Sex: {user_info['sex']}")
# #         st.write(f"Age: {user_info['age']}")

# #     except Exception as e:
# #         st.error(f"Error rendering dashboard: {e}")
    
# # def preprocess(text): 
# #     def remove_url(text):
# #         return re.sub(r"http\S+", "", text) 
# #     exclude = string.punctuation
# #     def remove_punctuation(text):
# #         return text.translate(str.maketrans("", "", exclude))
    
# #     def remove_stopwords(text):
# #         stopword = stopwords.words('english')
# #         new_text = []
# #         for word in text.split():
# #             if word in stopword:
# #                 new_text.append('')
# #             else:
# #                 new_text.append(word)
# #         x = new_text[:]
# #         new_text.clear()
# #         return " ".join(x)
# #     def lemmatize_text(text):
# #         words = word_tokenize(text)
# #         lemmatizer = WordNetLemmatizer()
# #         lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
# #         return " ".join(lemmatized_words)
# #     text = remove_url(text)
# #     text = remove_punctuation(text)
# #     text = remove_stopwords(text)
# #     text = lemmatize_text(text)
# #     return text
   
# # def transcribe_audio_from_data(file_data):
# #     try:
# #         with open("temp.mp3", "wb") as f:
# #             f.write(file_data)
# #         model = whisper.load_model("base")
# #         result = model.transcribe("temp.mp3")
# #         os.remove("temp.mp3")
# #         return result['text']
# #     except Exception as e:
# #         st.error(f"Error transcribing audio: {e}")
# #         return ""

# # def get_combined_diagnosis(voice_result, text_result):
# #     """Combine voice and text analysis results to get a final diagnosis"""
# #     # If voice model had an error or wasn't loaded
# #     if voice_result == "error" or voice_result == "model_not_loaded":
# #         return "Voice analysis unavailable, assessment incomplete", "info"
        
# #     # Define rules for combined diagnosis
# #     if voice_result == "depression" and text_result == "suicide":
# #         return "High risk - Suicidal tendencies detected", "danger"
# #     elif voice_result == "depression" and text_result == "depression":
# #         return "Moderate risk - Clinical depression likely", "warning"
# #     elif voice_result == "depression" and text_result == "normal":
# #         return "Low risk - Mild depression possible", "info"
# #     elif voice_result == "normal" and text_result == "depression":
# #         return "Low risk - Situational depression possible", "info"
# #     elif voice_result == "normal" and text_result == "suicide":
# #         return "Moderate risk - Suicidal ideation detected", "warning"
# #     elif voice_result == "normal" and text_result == "normal":
# #         return "No risk - Normal mental state", "success"
# #     else:
# #         return "Assessment inconclusive", "info"

# # def main(json_file_path="data.json"):
# #     st.sidebar.title("Mental Health Assessment")
# #     page = st.sidebar.radio(
# #         "Go to",
# #         ("Signup/Login", "Dashboard", "Mental Health Assessment"),
# #         key="mental_health_assessment",
# #     )

# #     if page == "Signup/Login":
# #         st.title("Signup/Login Page")
# #         login_or_signup = st.radio(
# #             "Select an option", ("Login", "Signup"), key="login_signup"
# #         )
# #         if login_or_signup == "Login":
# #             login(json_file_path)
# #         else:
# #             signup(json_file_path)

# #     elif page == "Dashboard":
# #         if session_state.get("logged_in"):
# #             render_dashboard(session_state["user_info"])
# #         else:
# #             st.warning("Please login/signup to view the dashboard.")

# #     elif page == "Mental Health Assessment":
# #         if session_state.get("logged_in"):
# #             st.title("Mental Health Assessment")
            
# #             # Initialize session state variables if they don't exist
# #             if "voice_result" not in st.session_state:
# #                 st.session_state.voice_result = None
# #                 st.session_state.voice_prob = 0.0
# #                 st.session_state.text_result = None
# #                 st.session_state.audio_data = None
# #                 st.session_state.analysis_complete = False
            
# #             # Voice Analysis Section
# #             st.header("Voice Analysis")
            
# #             # Check if models are loaded
# #             if not voice_model_loaded:
# #                 st.error("Voice analysis model could not be loaded. Some features will be limited.")
            
# #             options = ["Record", "Upload"]
# #             choice = st.radio("Choose an option for voice input", options)
            
# #             if choice == "Record":
# #                 st.write("Click the button below to start recording:")
# #                 audio = st_audiorec()
# #                 if audio is not None:
# #                     st.session_state.audio_data = audio
# #                     if st.button('Analyze Recording'):
# #                         # Save audio for analysis
# #                         with open("temp_analysis.wav", "wb") as f:
# #                             f.write(audio)
                        
# #                         # Analyze voice for depression
# #                         st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                        
# #                         # Initialize text result as voice result (fallback)
# #                         st.session_state.text_result = st.session_state.voice_result
                        
# #                         # Only attempt text analysis if the model was loaded properly
# #                         if text_model_loaded:
# #                             try:
# #                                 # Get transcribed text but don't display it
# #                                 transcribed_text = transcribe_audio_from_data(audio)
                                
# #                                 if transcribed_text:
# #                                     # Process text using the TF-IDF vectorizer that the model was trained with
# #                                     processed_text = preprocess(transcribed_text)
                                    
# #                                     # Use the vectorizer to transform the text
# #                                     tfidf_vector = vectorizer.transform([processed_text])
                                    
# #                                     # Check if features match what the model expects
# #                                     text_pred = text_model.predict(tfidf_vector)
# #                                     st.session_state.text_result = text_pred[0]
# #                             except Exception as e:
# #                                 # If there's any error in text analysis, silently fallback to voice only
# #                                 pass
                        
# #                         st.session_state.analysis_complete = True
                        
# #                         # Remove temporary file
# #                         if os.path.exists("temp_analysis.wav"):
# #                             os.remove("temp_analysis.wav")
            
# #             elif choice == "Upload":
# #                 audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
# #                 if audio is not None:
# #                     st.audio(audio, format="audio/wav")
# #                     st.session_state.audio_data = audio.read()
                    
# #                     if st.button('Analyze Upload'):
# #                         # Save audio for analysis
# #                         with open("temp_analysis.wav", "wb") as f:
# #                             f.write(st.session_state.audio_data)
                        
# #                         # Analyze voice for depression
# #                         st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                        
# #                         # Initialize text result as voice result (fallback)
# #                         st.session_state.text_result = st.session_state.voice_result
                        
# #                         # Only attempt text analysis if the model was loaded properly
# #                         if text_model_loaded:
# #                             try:
# #                                 # Get transcribed text but don't display it
# #                                 transcribed_text = transcribe_audio_from_data(st.session_state.audio_data)
                                
# #                                 if transcribed_text:
# #                                     # Process text using the TF-IDF vectorizer that the model was trained with
# #                                     processed_text = preprocess(transcribed_text)
                                    
# #                                     # Use the vectorizer to transform the text
# #                                     tfidf_vector = vectorizer.transform([processed_text])
                                    
# #                                     # Check if features match what the model expects
# #                                     text_pred = text_model.predict(tfidf_vector)
# #                                     st.write(text_pred)
# #                                     st.session_state.text_result = text_pred[0]
# #                             except Exception as e:
# #                                 # If there's any error in text analysis, silently fallback to voice only
# #                                 pass
                        
# #                         st.session_state.analysis_complete = True
                        
# #                         # Remove temporary file
# #                         if os.path.exists("temp_analysis.wav"):
# #                             os.remove("temp_analysis.wav")
            
# #             # Display results
# #             if st.session_state.analysis_complete:
# #                 st.header("Assessment Results")
                
# #                 # Display voice analysis results
# #                 st.subheader("Voice Analysis:")
# #                 if st.session_state.voice_result:
# #                     if st.session_state.voice_result == "depression":
# #                         st.warning(f"Voice indicates depression tendencies (Confidence: {st.session_state.voice_prob:.2f})")
# #                     elif st.session_state.voice_result == "normal":
# #                         st.success(f"Voice indicates normal mental state (Confidence: {1-st.session_state.voice_prob:.2f})")
# #                     elif st.session_state.voice_result == "model_not_loaded":
# #                         st.error("Voice analysis model not loaded properly")
# #                     else:
# #                         st.info("Voice analysis inconclusive")
                
# #                 # Display combined diagnosis
# #                 if st.session_state.voice_result:
# #                     # Get diagnosis
# #                     diagnosis, status = get_combined_diagnosis(
# #                         st.session_state.voice_result, 
# #                         st.session_state.text_result
# #                     )
                    
# #                     # Display diagnosis
# #                     st.subheader("Final Assessment:")
# #                     if status == "danger":
# #                         st.error(diagnosis)
# #                     elif status == "warning":
# #                         st.warning(diagnosis)
# #                     elif status == "success":
# #                         st.success(diagnosis)
# #                     else:
# #                         st.info(diagnosis)
                    
# #                     # Provide resources based on diagnosis
# #                     st.subheader("Resources:")
# #                     if "Suicidal" in diagnosis:
# #                         st.write("Some resources for controlling suicidal tendencies:")
# #                         st.markdown("- [Are you feeling Suicidal?](https://www.helpguide.org/articles/suicide-prevention/are-you-feeling-suicidal.htm)")
# #                         st.markdown("- [Suicide and Suicidal Thoughts](https://www.mayoclinic.org/diseases-conditions/suicide/symptoms-causes/syc-20378048)")
# #                         st.markdown("- [Suicide Prevention](https://www.ted.com/talks/ashleigh_husbands_suicide_prevention?hasSummary=true)")
# #                         st.markdown("- [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/)")
# #                         st.markdown("- [Crisis Text Line](https://www.crisistextline.org/)")
# #                         st.markdown("- [International Association for Suicide Prevention](https://www.iasp.info/resources/Crisis_Centres/)")
# #                     elif "depression" in diagnosis.lower():
# #                         st.write("Here are some resources to help cope with depression:")
# #                         st.markdown("- [How to cope up with depression](https://www.youtube.com/watch?v=8Su5VtKeXU8)")
# #                         st.markdown("- [Ways to cope with depression](https://www.medicalnewstoday.com/articles/327018)")
# #                         st.markdown("- [Depression Treatment](https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/depression-treatment-and-management)")
# #                         st.markdown("- [Don't suffer from Depression in Silence](https://www.ted.com/talks/nikki_webber_allen_don_t_suffer_from_your_depression_in_silence?hasSummary=true&language=en)")
# #                     elif "Normal" in diagnosis:
# #                         st.write("Great job! Keep up the positivity and stay clear of negative thinking.")
# #         else:
# #             st.warning("Please login/signup to use the app.")
            
# # if __name__ == "__main__":
# #     initialize_database()
# #     main()


# # import streamlit as st
# # import json
# # import os
# # import re
# # import string
# # from st_audiorec import st_audiorec
# # import io
# # import whisper
# # from nltk.corpus import stopwords
# # from nltk.tokenize import word_tokenize
# # from nltk.stem import WordNetLemmatizer
# # import speech_recognition as sr
# # from pydub import AudioSegment
# # import pickle
# # import numpy as np
# # import librosa
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # from sklearn.feature_extraction.text import TfidfVectorizer

# # # Initialize session state
# # session_state = st.session_state
# # if "user_index" not in st.session_state:
# #     st.session_state["user_index"] = 0

# # # Load text classification model
# # try:
# #     with open('logistic.pkl', 'rb') as f:
# #         text_model = pickle.load(f)

# #     with open('tfidf_vectorizer.pik', 'rb') as f:
# #         vectorizer = pickle.load(f)
# #     text_model_loaded = True
# # except Exception as e:
# #     st.error(f"Error loading text model: {e}")
# #     text_model_loaded = False
    
# # # Load voice depression detection model (MLP)
# # try:
# #     voice_model = load_model("mlp_model.keras")
# #     voice_model_loaded = True
# # except Exception as e:
# #     st.error(f"Error loading voice model: {e}")
# #     voice_model_loaded = False

# # # Helper functions for voice feature extraction
# # def noise(data):
# #     """Add random noise to the audio data"""
# #     noise_amp = 0.035*np.random.uniform()*np.amax(data)
# #     data = data + noise_amp*np.random.normal(size=data.shape[0])
# #     return data

# # def stretch(data, rate=0.8):
# #     """Stretch the audio data"""
# #     return librosa.effects.time_stretch(data, rate=rate)

# # def pitch(data, sampling_rate, pitch_factor=0.7):
# #     """Change the pitch of the audio data"""
# #     return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

# # def extract_features(data, sample_rate):
# #     """Extract audio features for depression detection"""
# #     # ZCR
# #     result = np.array([])
# #     zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
# #     result = np.hstack((result, zcr))  # stacking horizontally

# #     # Chroma_stft
# #     stft = np.abs(librosa.stft(data))
# #     chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, chroma_stft))  # stacking horizontally

# #     # MFCC
# #     mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, mfcc))  # stacking horizontally

# #     # Root Mean Square Value
# #     rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
# #     result = np.hstack((result, rms))  # stacking horizontally

# #     # MelSpectogram
# #     mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, mel))  # stacking horizontally

# #     # Spectral Contrast
# #     spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, spectral_contrast))

# #     # Tonnetz
# #     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, tonnetz))

# #     # Rolloff
# #     rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
# #     result = np.hstack((result, rolloff))

# #     # Zero Crossing Rate Variance
# #     zcr_var = np.var(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
# #     result = np.hstack((result, zcr_var))

# #     return result

# # def analyze_voice(audio_path, sample_rate=22050):
# #     """Analyze voice for depression detection using MLP model"""
# #     if not voice_model_loaded:
# #         return "model_not_loaded", 0.0
    
# #     try:
# #         # Extract features from the audio
# #         data, _ = librosa.load(audio_path, sr=sample_rate, duration=2.5, offset=0.6)
# #         features = extract_features(data, sample_rate)
        
# #         # Reshape features for model input
# #         features = np.array([features])
        
# #         # Make prediction using the MLP model
# #         prediction = voice_model.predict(features)
        
# #         # Get class with highest probability
# #         predicted_class = np.argmax(prediction[0])
# #         confidence = prediction[0][predicted_class]
        
# #         # Map prediction to label (0 is Depressed, 1 is Non-Depressed)
# #         if predicted_class == 0:
# #             return "depressed", confidence
# #         else:
# #             return "normal", confidence
# #     except Exception as e:
# #         st.error(f"Error analyzing voice: {e}")
# #         return "error", 0.0

# # def signup(json_file_path="data.json"):
# #     st.title("Signup Page")
# #     with st.form("signup_form"):
# #         st.write("Fill in the details below to create an account:")
# #         name = st.text_input("Name:")
# #         email = st.text_input("Email:")
# #         age = st.number_input("Age:", min_value=0, max_value=120)
# #         sex = st.radio("Sex:", ("Male", "Female", "Other"))
# #         password = st.text_input("Password:", type="password")
# #         confirm_password = st.text_input("Confirm Password:", type="password")

# #         if st.form_submit_button("Signup"):
# #             if password == confirm_password:
# #                 user = create_account(name, email, age, sex, password, json_file_path)
# #                 session_state["logged_in"] = True
# #                 session_state["user_info"] = user
# #             else:
# #                 st.error("Passwords do not match. Please try again.")

# # def check_login(username, password, json_file_path="data.json"):
# #     try:
# #         with open(json_file_path, "r") as json_file:
# #             data = json.load(json_file)

# #         for user in data["users"]:
# #             if user["email"] == username and user["password"] == password:
# #                 session_state["logged_in"] = True
# #                 session_state["user_info"] = user
# #                 st.success("Login successful!")
# #                 return user

# #         st.error("Invalid credentials. Please try again.")
# #         return None
# #     except Exception as e:
# #         st.error(f"Error checking login: {e}")
# #         return None

# # def initialize_database(json_file_path="data.json"):
# #     try:
# #         # Check if JSON file exists
# #         if not os.path.exists(json_file_path):
# #             # Create an empty JSON structure
# #             data = {"users": []}
# #             with open(json_file_path, "w") as json_file:
# #                 json.dump(data, json_file)
# #     except Exception as e:
# #         print(f"Error initializing database: {e}")
        
# # def create_account(name, email, age, sex, password, json_file_path="data.json"):
# #     try:
# #         # Check if the JSON file exists or is empty
# #         if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
# #             data = {"users": []}
# #         else:
# #             with open(json_file_path, "r") as json_file:
# #                 data = json.load(json_file)

# #         # Append new user data to the JSON structure
# #         user_info = {
# #             "name": name,
# #             "email": email,
# #             "age": age,
# #             "sex": sex,
# #             "password": password,
# #         }
# #         data["users"].append(user_info)

# #         # Save the updated data to JSON
# #         with open(json_file_path, "w") as json_file:
# #             json.dump(data, json_file, indent=4)

# #         st.success("Account created successfully! You can now login.")
# #         return user_info
# #     except json.JSONDecodeError as e:
# #         st.error(f"Error decoding JSON: {e}")
# #         return None
# #     except Exception as e:
# #         st.error(f"Error creating account: {e}")
# #         return None

# # def login(json_file_path="data.json"):
# #     st.title("Login Page")
# #     username = st.text_input("Email:")
# #     password = st.text_input("Password:", type="password")

# #     login_button = st.button("Login")

# #     if login_button:
# #         user = check_login(username, password, json_file_path)
# #         if user is not None:
# #             session_state["logged_in"] = True
# #             session_state["user_info"] = user
# #         else:
# #             st.error("Invalid credentials. Please try again.")

# # def get_user_info(email, json_file_path="data.json"):
# #     try:
# #         with open(json_file_path, "r") as json_file:
# #             data = json.load(json_file)
# #             for user in data["users"]:
# #                 if user["email"] == email:
# #                     return user
# #         return None
# #     except Exception as e:
# #         st.error(f"Error getting user information: {e}")
# #         return None

# # def render_dashboard(user_info, json_file_path="data.json"):
# #     try:
# #         st.title(f"Welcome to the Dashboard, {user_info['name']}!")
# #         st.subheader("User Information:")
# #         st.write(f"Name: {user_info['name']}")
# #         st.write(f"Sex: {user_info['sex']}")
# #         st.write(f"Age: {user_info['age']}")

# #     except Exception as e:
# #         st.error(f"Error rendering dashboard: {e}")
    
# # def preprocess(text): 
# #     def remove_url(text):
# #         return re.sub(r"http\S+", "", text) 
# #     exclude = string.punctuation
# #     def remove_punctuation(text):
# #         return text.translate(str.maketrans("", "", exclude))
    
# #     def remove_stopwords(text):
# #         stopword = stopwords.words('english')
# #         new_text = []
# #         for word in text.split():
# #             if word in stopword:
# #                 new_text.append('')
# #             else:
# #                 new_text.append(word)
# #         x = new_text[:]
# #         new_text.clear()
# #         return " ".join(x)
# #     def lemmatize_text(text):
# #         words = word_tokenize(text)
# #         lemmatizer = WordNetLemmatizer()
# #         lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
# #         return " ".join(lemmatized_words)
# #     text = remove_url(text)
# #     text = remove_punctuation(text)
# #     text = remove_stopwords(text)
# #     text = lemmatize_text(text)
# #     return text
   
# # def transcribe_audio_from_data(file_data):
# #     try:
# #         with open("temp.mp3", "wb") as f:
# #             f.write(file_data)
# #         model = whisper.load_model("base")
# #         result = model.transcribe("temp.mp3")
# #         os.remove("temp.mp3")
# #         return result['text']
# #     except Exception as e:
# #         st.error(f"Error transcribing audio: {e}")
# #         return ""

# # def get_combined_diagnosis(voice_result, text_result):
# #     """Combine voice and text analysis results according to the specified rules"""
# #     # Debug output
    
    
# #     # If voice model had an error or wasn't loaded
# #     if voice_result == "error" or voice_result == "model_not_loaded":
# #         return "Voice analysis unavailable, assessment incomplete", "info"
    
# #     # Handle "non-suicide" as equivalent to "normal" for text
# #     text_normal = text_result == "normal" or text_result == "non-suicide"
    
# #     # Implementation of the specified rules:
# #     # 1. depressed + depressed == depressed
# #     if voice_result == "depressed" and text_result == "depression":
# #         return "Depression detected", "warning"
    
# #     # 2. depressed + suicide = suicide
# #     elif voice_result == "depressed" and text_result == "suicide":
# #         return "Suicide risk detected", "danger"
    
# #     # 3. depressed + normal == normal
# #     elif voice_result == "depressed" and text_normal:
# #         return "Non-suicidal", "info"
    
# #     # 4. normal + normal == normal
# #     elif voice_result == "normal" and text_normal:
# #         return "Non-suicidal", "success"
    
# #     # 5. normal + suicide == depressed
# #     elif voice_result == "normal" and text_result == "suicide":
# #         return "Depression detected", "warning"
    
# #     # 6. normal + depressed = depressed
# #     elif voice_result == "normal" and text_result == "depression":
# #         return "Depression detected", "warning"
    
# #     # Default case
# #     else:
# #         return "Assessment inconclusive", "info"

# # def main(json_file_path="data.json"):
# #     st.sidebar.title("Mental Health Assessment")
# #     page = st.sidebar.radio(
# #         "Go to",
# #         ("Signup/Login", "Dashboard", "Mental Health Assessment"),
# #         key="mental_health_assessment",
# #     )

# #     if page == "Signup/Login":
# #         st.title("Signup/Login Page")
# #         login_or_signup = st.radio(
# #             "Select an option", ("Login", "Signup"), key="login_signup"
# #         )
# #         if login_or_signup == "Login":
# #             login(json_file_path)
# #         else:
# #             signup(json_file_path)

# #     elif page == "Dashboard":
# #         if session_state.get("logged_in"):
# #             render_dashboard(session_state["user_info"])
# #         else:
# #             st.warning("Please login/signup to view the dashboard.")

# #     elif page == "Mental Health Assessment":
# #         if session_state.get("logged_in"):
# #             st.title("Mental Health Assessment")
            
# #             # Initialize session state variables if they don't exist
# #             if "voice_result" not in st.session_state:
# #                 st.session_state.voice_result = None
# #                 st.session_state.voice_prob = 0.0
# #                 st.session_state.text_result = None
# #                 st.session_state.audio_data = None
# #                 st.session_state.analysis_complete = False
            
# #             # Voice Analysis Section
# #             st.header("Voice Analysis")
            
# #             # Check if models are loaded
# #             if not voice_model_loaded:
# #                 st.error("Voice analysis model could not be loaded. Some features will be limited.")
            
# #             options = ["Record", "Upload"]
# #             choice = st.radio("Choose an option for voice input", options)
            
# #             if choice == "Record":
# #                 st.write("Click the button below to start recording:")
# #                 audio = st_audiorec()
# #                 if audio is not None:
# #                     st.session_state.audio_data = audio
# #                     if st.button('Analyze Recording'):
# #                         # Save audio for analysis
# #                         with open("temp_analysis.wav", "wb") as f:
# #                             f.write(audio)
                        
# #                         # Analyze voice for depression
# #                         st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                        
# #                         # Initialize text result as voice result (fallback)
# #                         st.session_state.text_result = "normal"  # Default fallback
                        
# #                         # Only attempt text analysis if the model was loaded properly
# #                         if text_model_loaded:
# #                             try:
# #                                 # Get transcribed text but don't display it
# #                                 transcribed_text = transcribe_audio_from_data(audio)
                                
# #                                 if transcribed_text:
# #                                     # Process text using the TF-IDF vectorizer
# #                                     processed_text = preprocess(transcribed_text)
                                    
# #                                     # Use the vectorizer to transform the text
# #                                     tfidf_vector = vectorizer.transform([processed_text])
                                    
# #                                     # Get prediction from text model
# #                                     text_pred = text_model.predict(tfidf_vector)
# #                                     st.session_state.text_result = text_pred[0]
# #                             except Exception as e:
# #                                 # If there's any error in text analysis, silently fallback
# #                                 pass
                        
# #                         st.session_state.analysis_complete = True
                        
# #                         # Remove temporary file
# #                         if os.path.exists("temp_analysis.wav"):
# #                             os.remove("temp_analysis.wav")
            
# #             elif choice == "Upload":
# #                 audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
# #                 if audio is not None:
# #                     st.audio(audio, format="audio/wav")
# #                     st.session_state.audio_data = audio.read()
                    
# #                     if st.button('Analyze Upload'):
# #                         # Save audio for analysis
# #                         with open("temp_analysis.wav", "wb") as f:
# #                             f.write(st.session_state.audio_data)
                        
# #                         # Analyze voice for depression
# #                         st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                        
# #                         # Initialize text result as voice result (fallback)
# #                         st.session_state.text_result = "normal"  # Default fallback
                        
# #                         # Only attempt text analysis if the model was loaded properly
# #                         if text_model_loaded:
# #                             try:
# #                                 # Get transcribed text but don't display it
# #                                 transcribed_text = transcribe_audio_from_data(st.session_state.audio_data)
                                
# #                                 if transcribed_text:
# #                                     # Process text using the TF-IDF vectorizer
# #                                     processed_text = preprocess(transcribed_text)
                                    
# #                                     # Use the vectorizer to transform the text
# #                                     tfidf_vector = vectorizer.transform([processed_text])
                                    
# #                                     # Get prediction from text model
# #                                     text_pred = text_model.predict(tfidf_vector)
# #                                     st.session_state.text_result = text_pred[0]
# #                             except Exception as e:
# #                                 # If there's any error in text analysis, silently fallback
# #                                 pass
                        
# #                         st.session_state.analysis_complete = True
                        
# #                         # Remove temporary file
# #                         if os.path.exists("temp_analysis.wav"):
# #                             os.remove("temp_analysis.wav")
            
# #             # Display final combined results
# #             if st.session_state.analysis_complete:
# #                 # Get the combined diagnosis
# #                 diagnosis, status = get_combined_diagnosis(
# #                     st.session_state.voice_result, 
# #                     st.session_state.text_result
# #                 )
                
# #                 # Display diagnosis
# #                 st.header("Assessment Results")
# #                 if status == "danger":
# #                     st.error(diagnosis)
# #                 elif status == "warning":
# #                     st.warning(diagnosis)
# #                 elif status == "success":
# #                     st.success(diagnosis)
# #                 else:
# #                     st.info(diagnosis)
                
# #                 # Provide resources based on diagnosis
# #                 st.subheader("Resources:")
# #                 if "Suicide" in diagnosis:
# #                     st.write("Some resources for controlling suicidal tendencies:")
# #                     st.markdown("- [Are you feeling Suicidal?](https://www.helpguide.org/articles/suicide-prevention/are-you-feeling-suicidal.htm)")
# #                     st.markdown("- [Suicide and Suicidal Thoughts](https://www.mayoclinic.org/diseases-conditions/suicide/symptoms-causes/syc-20378048)")
# #                     st.markdown("- [Suicide Prevention](https://www.ted.com/talks/ashleigh_husbands_suicide_prevention?hasSummary=true)")
# #                     st.markdown("- [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/)")
# #                     st.markdown("- [Crisis Text Line](https://www.crisistextline.org/)")
# #                     st.markdown("- [International Association for Suicide Prevention](https://www.iasp.info/resources/Crisis_Centres/)")
# #                 elif "Depression" in diagnosis:
# #                     st.write("Here are some resources to help cope with depression:")
# #                     st.markdown("- [How to cope up with depression](https://www.youtube.com/watch?v=8Su5VtKeXU8)")
# #                     st.markdown("- [Ways to cope with depression](https://www.medicalnewstoday.com/articles/327018)")
# #                     st.markdown("- [Depression Treatment](https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/depression-treatment-and-management)")
# #                     st.markdown("- [Don't suffer from Depression in Silence](https://www.ted.com/talks/nikki_webber_allen_don_t_suffer_from_your_depression_in_silence?hasSummary=true&language=en)")
# #                 else:
# #                     st.write("Great job! Keep up the positivity and stay clear of negative thinking.")
# #         else:
# #             st.warning("Please login/signup to use the app.")
            
# # if __name__ == "__main__":
# #     initialize_database()
# #     main()


# import streamlit as st
# import json
# import os
# import re
# import string
# from st_audiorec import st_audiorec
# import io
# import whisper
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import speech_recognition as sr
# from pydub import AudioSegment
# import pickle
# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.feature_extraction.text import TfidfVectorizer
# import base64
# from PIL import Image
# from io import BytesIO
# import time
# import random
# from datetime import datetime

# # Set page configuration
# st.set_page_config(
#     page_title="MindWell - Mental Health Assessment",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Custom CSS to improve aesthetics
# def local_css():
#     st.markdown("""
#     <style>
#         .main {
#             background-color: #f8f9fa;
#         }
#         .stButton>button {
#             background-color: #6a0dad;
#             color: white;
#             border-radius: 10px;
#             border: none;
#             padding: 10px 24px;
#             font-weight: bold;
#             transition: all 0.3s ease;
#         }
#         .stButton>button:hover {
#             background-color: #4a0a8c;
#             transform: translateY(-2px);
#             box-shadow: 0 5px 15px rgba(0,0,0,0.1);
#         }
#         .logout-btn {
#             background-color: #ff5252 !important;
#         }
#         .logout-btn:hover {
#             background-color: #ff1a1a !important;
#         }
#         .result-card {
#             background-color: white;
#             border-radius: 10px;
#             padding: 20px;
#             margin: 20px 0;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#         }
#         h1, h2, h3 {
#             color: #333;
#             font-family: 'Helvetica Neue', Helvetica, sans-serif;
#         }
#         .resource-card {
#             background-color: #f1f5f9;
#             border-radius: 10px;
#             padding: 15px;
#             margin: 10px 0;
#             border-left: 5px solid #6a0dad;
#         }
#         .css-18e3th9 {
#             padding-top: 2rem;
#         }
#         .css-1d391kg {
#             padding-top: 3.5rem;
#         }
#         .sidebar .sidebar-content {
#             background-image: linear-gradient(#f8f9fa,#e9ecef);
#         }
#         /* Custom card styling */
#         .card {
#             border-radius: 10px;
#             border: 1px solid #e0e0e0;
#             padding: 1.5rem;
#             margin-bottom: 1rem;
#             background-color: white;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             transition: transform 0.3s ease;
#         }
#         .card:hover {
#             transform: translateY(-5px);
#             box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
#         }
#         /* Status colors */
#         .status-danger {
#             border-left: 5px solid #ff4b4b !important;
#             background-color: rgba(255, 75, 75, 0.05);
#         }
#         .status-warning {
#             border-left: 5px solid #ffa62b !important;
#             background-color: rgba(255, 166, 43, 0.05);
#         }
#         .status-success {
#             border-left: 5px solid #0ead69 !important;
#             background-color: rgba(14, 173, 105, 0.05);
#         }
#         .status-info {
#             border-left: 5px solid #4361ee !important;
#             background-color: rgba(67, 97, 238, 0.05);
#         }
#         .resource-title {
#             font-weight: bold;
#             color: #6a0dad;
#         }
#         /* Recommendation Section */
#         .recommendation-section {
#             margin-top: 20px;
#             padding: 20px;
#             border-radius: 10px;
#             background-color: #f8f9fa;
#         }
#         .recommendation-card {
#             background-color: white;
#             border-radius: 8px;
#             padding: 15px;
#             margin-bottom: 15px;
#             box-shadow: 0 2px 5px rgba(0,0,0,0.05);
#             border-left: 4px solid #6a0dad;
#         }
#         .recommendation-title {
#             font-weight: bold;
#             color: #6a0dad;
#             display: flex;
#             align-items: center;
#             gap: 8px;
#         }
#         /* Emoji styling */
#         .emoji {
#             font-size: 1.5em;
#             margin-right: 0.5em;
#             display: inline-block;
#         }
#         /* Section headers with gradients */
#         .gradient-header {
#             background: linear-gradient(90deg, #6a0dad, #9b59b6);
#             color: white !important;
#             padding: 10px 20px;
#             border-radius: 10px;
#             margin-bottom: 20px;
#             text-align: center;
#         }
#         /* Animation for assessment in progress */
#         @keyframes pulse {
#             0% {
#                 transform: scale(1);
#                 opacity: 1;
#             }
#             50% {
#                 transform: scale(1.05);
#                 opacity: 0.8;
#             }
#             100% {
#                 transform: scale(1);
#                 opacity: 1;
#             }
#         }
#         .pulse-animation {
#             animation: pulse 1.5s infinite;
#         }
#         /* Progress bar styling */
#         .stProgress > div > div > div {
#             background-color: #6a0dad;
#         }
#         /* Hide hamburger menu */
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#     </style>
#     """, unsafe_allow_html=True)

# # Apply CSS
# local_css()

# # Function to create a custom header with emoji
# def custom_header(text, emoji, level=2):
#     emoji_span = f'<span class="emoji">{emoji}</span>'
#     if level == 1:
#         st.markdown(f'<h1>{emoji_span} {text}</h1>', unsafe_allow_html=True)
#     elif level == 2:
#         st.markdown(f'<h2>{emoji_span} {text}</h2>', unsafe_allow_html=True)
#     elif level == 3:
#         st.markdown(f'<h3>{emoji_span} {text}</h3>', unsafe_allow_html=True)

# # Function to load and display the logo
# def display_logo():
#     # Create a simple logo with PIL if no logo file exists
#     img = Image.new('RGB', (200, 200), color = (106, 13, 173))
    
#     # Add some basic text to the image
#     from PIL import ImageDraw, ImageFont
#     draw = ImageDraw.Draw(img)
    
#     # Try to create a circular shape
#     draw.ellipse((10, 10, 190, 190), fill=(255, 255, 255))
#     draw.ellipse((20, 20, 180, 180), fill=(106, 13, 173))
    
#     # Try to add the letter M
#     try:
#         # This will work if a suitable font is available
#         font = ImageFont.truetype("arial.ttf", 120)
#         draw.text((60, 40), "M", fill=(255, 255, 255), font=font)
#     except:
#         # Fallback if font is not available
#         try:
#             font = ImageFont.load_default()
#             draw.text((80, 80), "M", fill=(255, 255, 255), font=font)
#         except:
#             pass
    
#     d = BytesIO()
#     img.save(d, 'PNG')
    
#     # Create a colored background with the app name
#     st.markdown(f"""
#     <div style="display: flex; align-items: center; background: linear-gradient(90deg, #6a0dad, #9b59b6); padding: 1rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
#         <img src="data:image/png;base64,{base64.b64encode(d.getvalue()).decode()}" style="width: 80px; height: 80px; border-radius: 50%; margin-right: 20px; background-color: white; padding: 5px;">
#         <div>
#             <h1 style="margin: 0; color: white; font-family: 'Helvetica Neue', Helvetica, sans-serif;">MindWell 🧠</h1>
#             <p style="margin: 0; color: rgba(255,255,255,0.8);">Mental Health Assessment Platform</p>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# # Initialize session state
# if "user_index" not in st.session_state:
#     st.session_state["user_index"] = 0
# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False
# if "user_info" not in st.session_state:
#     st.session_state["user_info"] = None
# if "voice_result" not in st.session_state:
#     st.session_state.voice_result = None
#     st.session_state.voice_prob = 0.0
#     st.session_state.text_result = None
#     st.session_state.audio_data = None
#     st.session_state.analysis_complete = False
#     st.session_state.show_results = False
#     st.session_state.current_diagnosis = None
#     st.session_state.current_status = None

# # Load text classification model
# try:
#     with open('logistic.pkl', 'rb') as f:
#         text_model = pickle.load(f)

#     with open('tfidf_vectorizer.pik', 'rb') as f:
#         vectorizer = pickle.load(f)
#     text_model_loaded = True
# except Exception as e:
#     text_model_loaded = False
    
# # Load voice depression detection model (MLP)
# try:
#     voice_model = load_model("mlp_model.keras")
#     voice_model_loaded = True
# except Exception as e:
#     voice_model_loaded = False

# # Helper functions for voice feature extraction
# def noise(data):
#     """Add random noise to the audio data"""
#     noise_amp = 0.035*np.random.uniform()*np.amax(data)
#     data = data + noise_amp*np.random.normal(size=data.shape[0])
#     return data

# def stretch(data, rate=0.8):
#     """Stretch the audio data"""
#     return librosa.effects.time_stretch(data, rate=rate)

# def pitch(data, sampling_rate, pitch_factor=0.7):
#     """Change the pitch of the audio data"""
#     return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

# def extract_features(data, sample_rate):
#     """Extract audio features for depression detection"""
#     # ZCR
#     result = np.array([])
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
#     result = np.hstack((result, zcr))  # stacking horizontally

#     # Chroma_stft
#     stft = np.abs(librosa.stft(data))
#     chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, chroma_stft))  # stacking horizontally

#     # MFCC
#     mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, mfcc))  # stacking horizontally

#     # Root Mean Square Value
#     rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
#     result = np.hstack((result, rms))  # stacking horizontally

#     # MelSpectogram
#     mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, mel))  # stacking horizontally

#     # Spectral Contrast
#     spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, spectral_contrast))

#     # Tonnetz
#     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
#     result = np.hstack((result, tonnetz))

#     # Rolloff
#     rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, rolloff))

#     # Zero Crossing Rate Variance
#     zcr_var = np.var(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
#     result = np.hstack((result, zcr_var))

#     return result

# def analyze_voice(audio_path, sample_rate=22050):
#     """Analyze voice for depression detection using MLP model"""
#     if not voice_model_loaded:
#         return "model_not_loaded", 0.0
    
#     try:
#         # Extract features from the audio
#         data, _ = librosa.load(audio_path, sr=sample_rate, duration=2.5, offset=0.6)
#         features = extract_features(data, sample_rate)
        
#         # Reshape features for model input
#         features = np.array([features])
        
#         # Make prediction using the MLP model
#         prediction = voice_model.predict(features)
        
#         # Get class with highest probability
#         predicted_class = np.argmax(prediction[0])
#         confidence = prediction[0][predicted_class]
        
#         # Map prediction to label (0 is Depressed, 1 is Non-Depressed)
#         if predicted_class == 0:
#             return "depressed", confidence
#         else:
#             return "normal", confidence
#     except Exception as e:
#         return "error", 0.0

# def logout():
#     """Function to handle user logout"""
#     for key in list(st.session_state.keys()):
#         if key != "user_index":  # Keep this for consistent UI
#             del st.session_state[key]
#     st.session_state["logged_in"] = False
#     st.session_state["user_info"] = None
#     st.session_state.voice_result = None
#     st.session_state.voice_prob = 0.0
#     st.session_state.text_result = None
#     st.session_state.audio_data = None
#     st.session_state.analysis_complete = False
#     st.session_state.show_results = False
#     st.rerun()

# def signup(json_file_path="data.json"):
#     """Handle user signup"""
#     st.markdown("<h2 class='gradient-header'>✨ Create Your Account</h2>", unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1,2,1])
#     with col2:
#         with st.form("signup_form", border=False):
#             st.markdown("<p style='text-align: center;'>Fill in the details below to start your mental wellness journey:</p>", unsafe_allow_html=True)
#             name = st.text_input("🙋 Full Name")
#             email = st.text_input("📧 Email Address")
            
#             col_age, col_sex = st.columns(2)
#             with col_age:
#                 age = st.number_input("🎂 Age", min_value=18, max_value=120, value=25)
#             with col_sex:
#                 sex = st.selectbox("⚧ Gender", ("Male", "Female", "Other"))
            
#             password = st.text_input("🔒 Password", type="password")
#             confirm_password = st.text_input("🔒 Confirm Password", type="password")

#             st.markdown("<br>", unsafe_allow_html=True)
#             submitted = st.form_submit_button("✅ Create Account", use_container_width=True)
#             st.markdown("""
#             <p style='text-align: center; font-size: 0.8em; margin-top: 10px;'>
#                 By creating an account, you agree to our Terms of Service and Privacy Policy.
#             </p>
#             """, unsafe_allow_html=True)

#             if submitted:
#                 if password == confirm_password:
#                     user = create_account(name, email, age, sex, password, json_file_path)
#                     st.session_state["logged_in"] = True
#                     st.session_state["user_info"] = user
#                     st.success("🎉 Account created successfully! Welcome to MindWell.")
#                     st.rerun()
#                 else:
#                     st.error("❌ Passwords do not match. Please try again.")

# def check_login(username, password, json_file_path="data.json"):
#     """Verify user login credentials"""
#     try:
#         with open(json_file_path, "r") as json_file:
#             data = json.load(json_file)

#         for user in data["users"]:
#             if user["email"] == username and user["password"] == password:
#                 st.session_state["logged_in"] = True
#                 st.session_state["user_info"] = user
#                 return user

#         return None
#     except Exception as e:
#         return None

# def initialize_database(json_file_path="data.json"):
#     """Initialize the JSON database if it doesn't exist"""
#     try:
#         # Check if JSON file exists
#         if not os.path.exists(json_file_path):
#             # Create an empty JSON structure
#             data = {"users": []}
#             with open(json_file_path, "w") as json_file:
#                 json.dump(data, json_file)
#     except Exception as e:
#         print(f"Error initializing database: {e}")
        
# def create_account(name, email, age, sex, password, json_file_path="data.json"):
#     """Create a new user account"""
#     try:
#         # Check if the JSON file exists or is empty
#         if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
#             data = {"users": []}
#         else:
#             with open(json_file_path, "r") as json_file:
#                 data = json.load(json_file)

#         # Append new user data to the JSON structure
#         user_info = {
#             "name": name,
#             "email": email,
#             "age": age,
#             "sex": sex,
#             "password": password,
#             "assessment_history": []
#         }
#         data["users"].append(user_info)

#         # Save the updated data to JSON
#         with open(json_file_path, "w") as json_file:
#             json.dump(data, json_file, indent=4)

#         return user_info
#     except json.JSONDecodeError as e:
#         st.error(f"Error decoding JSON: {e}")
#         return None
#     except Exception as e:
#         st.error(f"Error creating account: {e}")
#         return None

# def login(json_file_path="data.json"):
#     """Handle user login"""
#     st.markdown("<h2 class='gradient-header'>👋 Welcome Back</h2>", unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1,2,1])
#     with col2:
#         with st.form("login_form", border=False):
#             st.markdown("<p style='text-align: center;'>Sign in to continue your mental wellness journey</p>", unsafe_allow_html=True)
#             username = st.text_input("📧 Email Address")
#             password = st.text_input("🔒 Password", type="password")

#             st.markdown("<br>", unsafe_allow_html=True)
#             submitted = st.form_submit_button("🔑 Login", use_container_width=True)

#             if submitted:
#                 user = check_login(username, password, json_file_path)
#                 if user is not None:
#                     st.success("✅ Login successful! Welcome back.")
#                     st.rerun()
#                 else:
#                     st.error("❌ Invalid email or password. Please try again.")

# def get_user_info(email, json_file_path="data.json"):
#     """Get user information from the database"""
#     try:
#         with open(json_file_path, "r") as json_file:
#             data = json.load(json_file)
#             for user in data["users"]:
#                 if user["email"] == email:
#                     return user
#         return None
#     except Exception as e:
#         return None

# def save_assessment_result(user_email, result, status, json_file_path="data.json"):
#     """Save assessment result to user history"""
#     try:
#         # Read the current data
#         with open(json_file_path, "r") as json_file:
#             data = json.load(json_file)
        
#         # Find the user
#         for user in data["users"]:
#             if user["email"] == user_email:
#                 # Initialize assessment_history if it doesn't exist
#                 if "assessment_history" not in user:
#                     user["assessment_history"] = []
                
#                 # Add the new assessment result with timestamp
#                 assessment_entry = {
#                     "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "result": result,
#                     "status": status
#                 }
                
#                 # Only keep the most recent 5 entries
#                 user["assessment_history"] = [assessment_entry] + user["assessment_history"][:4]
#                 break
        
#         # Save the updated data
#         with open(json_file_path, "w") as json_file:
#             json.dump(data, json_file, indent=4)
            
#     except Exception as e:
#         print(f"Error saving assessment result: {e}")

# def update_user_info():
#     """Update user info in session state with latest from database"""
#     if st.session_state.get("logged_in") and st.session_state.get("user_info"):
#         user_email = st.session_state["user_info"]["email"]
#         updated_info = get_user_info(user_email)
#         if updated_info:
#             st.session_state["user_info"] = updated_info

# def render_dashboard(user_info, json_file_path="data.json"):
#     """Render user dashboard"""
#     update_user_info()  # Make sure we have latest user data
    
#     # User welcome card
#     st.markdown(f"""
#     <div class="card">
#         <h2 style="color: #6a0dad; margin-bottom: 20px;">👋 Welcome, {user_info['name']}!</h2>
#         <div style="display: flex; gap: 20px; flex-wrap: wrap;">
#             <div style="flex: 1; min-width: 250px;">
#                 <h3>👤 Personal Information</h3>
#                 <p><strong>Name:</strong> {user_info['name']}</p>
#                 <p><strong>Email:</strong> {user_info['email']}</p>
#                 <p><strong>Age:</strong> {user_info['age']}</p>
#                 <p><strong>Gender:</strong> {user_info['sex']}</p>
#             </div>
#             <div style="flex: 1; min-width: 250px;">
#                 <h3>💡 Mental Wellness Tips</h3>
#                 <ul>
#                     <li>🧘 Practice mindfulness for 10 minutes daily</li>
#                     <li>☕ Take short breaks when feeling overwhelmed</li>
#                     <li>👥 Connect with loved ones regularly</li>
#                     <li>😴 Maintain a consistent sleep schedule</li>
#                     <li>🚶 Stay physically active with activities you enjoy</li>
#                 </ul>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Mental wellness resources
#     st.markdown("""
#     <div class="card">
#         <h3 style="color: #6a0dad;">🌟 Mental Wellness Resources</h3>
#         <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px;">
#             <div class="resource-card" style="flex: 1; min-width: 200px;">
#                 <h4 class="resource-title">📱 Meditation Apps</h4>
#                 <ul>
#                     <li>Headspace</li>
#                     <li>Calm</li>
#                     <li>Insight Timer</li>
#                 </ul>
#             </div>
#             <div class="resource-card" style="flex: 1; min-width: 200px;">
#                 <h4 class="resource-title">🌱 Self-Care Activities</h4>
#                 <ul>
#                     <li>Journaling daily thoughts</li>
#                     <li>Nature walks and fresh air</li>
#                     <li>Creative expression through art</li>
#                 </ul>
#             </div>
#             <div class="resource-card" style="flex: 1; min-width: 200px;">
#                 <h4 class="resource-title">👥 Support Resources</h4>
#                 <ul>
#                     <li>Local support groups</li>
#                     <li>Online communities for connection</li>
#                     <li>Professional therapy options</li>
#                 </ul>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Inspirational quote card - randomly selected
#     quotes = [
#         {"text": "You don't have to be positive all the time. It's perfectly okay to feel sad, angry, annoyed, frustrated, scared and anxious. Having feelings doesn't make you a negative person. It makes you human.", "author": "Lori Deschene"},
#         {"text": "Mental health problems don't define who you are. They are something you experience, but they are not you.", "author": "Unknown"},
#         {"text": "You are not alone in this. You have more support than you could ever know.", "author": "Unknown"},
#         {"text": "Your mental health is a priority. Your happiness is essential. Your self-care is a necessity.", "author": "Unknown"},
#         {"text": "Recovery is not one and done. It is a lifelong journey that takes place one day, one step at a time.", "author": "Unknown"}
#     ]
    
#     random_quote = random.choice(quotes)
    
#     st.markdown(f"""
#     <div class="card" style="background-color: #f0e6ff; border-left: 5px solid #6a0dad;">
#         <div style="font-style: italic; font-size: 1.1em; margin-bottom: 10px;">"{random_quote["text"]}"</div>
#         <div style="text-align: right; font-weight: bold;">— {random_quote["author"]}</div>
#     </div>
#     """, unsafe_allow_html=True)
    
# def preprocess(text): 
#     """Preprocess text for depression analysis"""
#     def remove_url(text):
#         return re.sub(r"http\S+", "", text) 
#     exclude = string.punctuation
#     def remove_punctuation(text):
#         return text.translate(str.maketrans("", "", exclude))
    
#     def remove_stopwords(text):
#         stopword = stopwords.words('english')
#         new_text = []
#         for word in text.split():
#             if word in stopword:
#                 new_text.append('')
#             else:
#                 new_text.append(word)
#         x = new_text[:]
#         new_text.clear()
#         return " ".join(x)
#     def lemmatize_text(text):
#         words = word_tokenize(text)
#         lemmatizer = WordNetLemmatizer()
#         lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
#         return " ".join(lemmatized_words)
#     text = remove_url(text)
#     text = remove_punctuation(text)
#     text = remove_stopwords(text)
#     text = lemmatize_text(text)
#     return text
   
# def transcribe_audio_from_data(file_data):
#     """Transcribe audio to text"""
#     try:
#         with open("temp.mp3", "wb") as f:
#             f.write(file_data)
#         model = whisper.load_model("base")
#         result = model.transcribe("temp.mp3")
#         os.remove("temp.mp3")
#         return result['text']
#     except Exception as e:
#         return ""

# def get_combined_diagnosis(voice_result, text_result):
#     """Combine voice and text analysis results according to the specified rules"""

#     if voice_result == "error" or voice_result == "model_not_loaded":
#         return "Voice analysis unavailable, assessment incomplete", "info"
    

#     text_normal = text_result == "normal" or text_result == "non-suicide"
    

#     if voice_result == "depressed" and text_result == "depression":
#         return "Depression detected", "warning"
    
  
#     elif voice_result == "depressed" and text_result == "suicide":
#         return "Suicide risk detected", "danger"
    

#     elif voice_result == "depressed" and text_normal:
#         return "Non-suicidal", "info"
    

#     elif voice_result == "normal" and text_normal:
#         return "Non-suicidal", "success"
    

#     elif voice_result == "normal" and text_result == "suicide":
#         return "Depression detected", "warning"
    

#     elif voice_result == "normal" and text_result == "depression":
#         return "Depression detected", "warning"
    
#     # Default case
#     else:
#         return "Assessment inconclusive", "info"

# def get_recommendations(diagnosis):
#     """Get recommendations based on the diagnosis"""
    
#     # Common recommendations for all mental health conditions
#     common_recs = [
#         "Practice mindfulness meditation daily",
#         "Maintain a regular sleep schedule",
#         "Stay physically active with activities you enjoy",
#         "Connect with supportive friends and family",
#         "Limit consumption of news and social media"
#     ]
    
#     # Specific recommendations based on diagnosis
#     if "Suicide risk" in diagnosis:
#         specific_recs = [
#             "Contact a crisis helpline immediately (988 or 1-800-273-8255)",
#             "Go to your nearest emergency room if thoughts are severe",
#             "Remove access to potential means of self-harm",
#             "Don't be alone - stay with someone you trust",
#             "Work with a professional to create a safety plan"
#         ]
#         emoji = "🚨"
#         color = "#ff4b4b"
#     elif "Depression" in diagnosis:
#         specific_recs = [
#             "Schedule an appointment with a mental health professional",
#             "Consider starting a mood journal to track patterns",
#             "Set small, achievable daily goals",
#             "Expose yourself to natural sunlight daily",
#             "Practice gratitude by noting three positive things each day"
#         ]
#         emoji = "💜"
#         color = "#ffa62b"
#     elif "Non-suicidal" in diagnosis:
#         specific_recs = [
#             "Continue building healthy mental habits",
#             "Explore new activities that bring you joy",
#             "Learn stress management techniques",
#             "Consider regular check-ins with a mental health professional",
#             "Share your wellness journey with others"
#         ]
#         emoji = "✨"
#         color = "#0ead69"
#     else:  # Assessment inconclusive
#         specific_recs = [
#             "Consider a more comprehensive assessment with a professional",
#             "Keep a mood journal to track your emotions",
#             "Learn more about different mental health conditions",
#             "Try the assessment again when you're feeling different",
#             "Focus on general wellness practices"
#         ]
#         emoji = "🔍"
#         color = "#4361ee"
    
#     return {
#         "common": common_recs,
#         "specific": specific_recs,
#         "emoji": emoji,
#         "color": color
#     }

# def display_results_and_recommendations(diagnosis, status):
#     """Display assessment results and recommendations"""
#     # Set status-specific styling
#     if status == "danger":
#         status_class = "status-danger"
#         emoji = "🚨"
#     elif status == "warning":
#         status_class = "status-warning"
#         emoji = "⚠️"
#     elif status == "success":
#         status_class = "status-success"
#         emoji = "✅"
#     else:
#         status_class = "status-info"
#         emoji = "ℹ️"
    
#     # Display the result card
#     st.markdown(f"""
#     <div class="card {status_class}">
#         <h2 style="display:flex;align-items:center;gap:10px;">{emoji} Assessment Result</h2>
#         <p style="font-size: 1.5em; font-weight: bold; margin: 20px 0;">{diagnosis}</p>
#         <p style="font-style: italic;">Assessment completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Get recommendations
#     recs = get_recommendations(diagnosis)
    
    
    
    
#     # Display specific recommendations
#     for rec in recs['specific']:
#         st.markdown(f"""
#         <div style="background-color: {recs['color']}15; border-left: 3px solid {recs['color']}; 
#                     padding: 15px; border-radius: 5px; flex: 1; min-width: 250px;">
#             <p style="margin: 0;">{rec}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("""
#             </div>
#         </div>
        
#         <div style="margin-top: 20px;">
#             <h3>General Wellness Practices</h3>
#             <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
#     """, unsafe_allow_html=True)
    
#     # Display common recommendations
#     for rec in recs['common']:
#         st.markdown(f"""
#         <div style="background-color: #f1f5f9; border-left: 3px solid #6a0dad; 
#                     padding: 15px; border-radius: 5px; flex: 1; min-width: 250px;">
#             <p style="margin: 0;">{rec}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("""
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Resources section
#     st.markdown("""
#     <div class="card">
#         <h2 style="display:flex;align-items:center;gap:10px;">📚 Helpful Resources</h2>
#     """, unsafe_allow_html=True)
    
#     # Resources based on diagnosis
#     if "Suicide risk" in diagnosis:
#         st.markdown("""
#         <div style="margin-top: 15px;">
#             <h3>Crisis Resources</h3>
#             <ul>
#                 <li><strong><a href="https://988lifeline.org/" target="_blank">988 Suicide & Crisis Lifeline</a></strong> - Call or text 988</li>
#                 <li><strong><a href="https://www.crisistextline.org/" target="_blank">Crisis Text Line</a></strong> - Text HOME to 741741</li>
#                 <li><strong><a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank">International Association for Suicide Prevention</a></strong> - Global resources</li>
#                 <li><strong><a href="https://www.betterhelp.com/" target="_blank">BetterHelp</a></strong> - Online counseling</li>
#                 <li><strong><a href="https://suicidepreventionlifeline.org/wp-content/uploads/2016/08/Brown_StanleySafetyPlanTemplate.pdf" target="_blank">Safety Plan Template</a></strong> - Downloadable safety plan</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
#     elif "Depression" in diagnosis:
#         st.markdown("""
#         <div style="margin-top: 15px;">
#             <h3>Depression Resources</h3>
#             <ul>
#                 <li><strong><a href="https://www.dbsalliance.org/" target="_blank">Depression and Bipolar Support Alliance</a></strong> - Support groups and resources</li>
#                 <li><strong><a href="https://www.nimh.nih.gov/health/topics/depression" target="_blank">National Institute of Mental Health</a></strong> - Information about depression</li>
#                 <li><strong><a href="https://www.psychologytoday.com/us/therapists" target="_blank">Psychology Today Therapist Finder</a></strong> - Find a therapist near you</li>
#                 <li><strong><a href="https://www.betterhelp.com/" target="_blank">BetterHelp</a></strong> - Online counseling</li>
#                 <li><strong><a href="https://www.headspace.com/" target="_blank">Headspace</a></strong> - Meditation app</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div style="margin-top: 15px;">
#             <h3>Mental Wellness Resources</h3>
#             <ul>
#                 <li><strong><a href="https://www.nami.org/" target="_blank">National Alliance on Mental Illness</a></strong> - Mental health education and support</li>
#                 <li><strong><a href="https://www.mhanational.org/" target="_blank">Mental Health America</a></strong> - Mental health resources</li>
#                 <li><strong><a href="https://www.headspace.com/" target="_blank">Headspace</a></strong> - Meditation app</li>
#                 <li><strong><a href="https://www.calm.com/" target="_blank">Calm</a></strong> - Sleep and meditation app</li>
#                 <li><strong><a href="https://www.betterhelp.com/" target="_blank">BetterHelp</a></strong> - Online counseling</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     # Disclaimer
#     st.markdown("""
#     <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; font-size: 0.9em;">
#         <p><strong>Disclaimer:</strong> This assessment is not a clinical diagnosis. If you're experiencing severe symptoms, 
#         please consult with a qualified mental health professional or go to your nearest emergency room.</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Save this assessment to user history
#     if st.session_state.get("logged_in") and st.session_state.get("user_info"):
#         save_assessment_result(
#             st.session_state["user_info"]["email"], 
#             diagnosis,
#             status
#         )

# def main(json_file_path="data.json"):
#     # Display logo
#     display_logo()
    
#     # Create sidebar
#     st.sidebar.markdown("<h2 style='text-align: center;'>🧠 MindWell</h2>", unsafe_allow_html=True)
    
#     # Add logout button if logged in
#     if st.session_state.get("logged_in"):
#         st.sidebar.button("📤 Logout", on_click=logout, key="logout_button", type="primary", 
#                          help="Click to log out of your account")
#         st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    
#     # Navigation menu
#     page = st.sidebar.radio(
#         "📋 Navigation",
#         ("👤 Account", "📊 Dashboard", "🔍 Mental Health Assessment"),
#         key="navigation",
#     )
    
#     # Add sidebar footer
#     st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
#     st.sidebar.markdown("""
#     <div style='text-align: center; padding: 10px; background-color: #f1f5f9; border-radius: 10px;'>
#         <p style='font-size: 0.8em; margin-bottom: 5px;'>MindWell Health Assessment</p>
#         <p style='font-size: 0.7em; color: #666;'>Version 2.0</p>
#     </div>
#     """, unsafe_allow_html=True)

#     # Page content
#     if page == "👤 Account":
#         if st.session_state.get("logged_in"):
#             st.markdown("<h2 class='gradient-header'>👤 Account Settings</h2>", unsafe_allow_html=True)
#             st.markdown("""
#             <div class="card">
#                 <h3>Your Profile</h3>
#                 <p>You are currently logged in. Use the navigation menu to access different features.</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Logout button on the page as well
#             st.button("📤 Logout from MindWell", on_click=logout, key="logout_button_page", 
#                      type="primary", help="Click to log out of your account")
#         else:
#             st.title("Welcome to MindWell")
#             login_or_signup = st.radio(
#                 "Choose an option:",
#                 ("🔑 Login", "✨ Signup"),
#                 key="login_signup",
#                 horizontal=True
#             )
#             if login_or_signup == "🔑 Login":
#                 login(json_file_path)
#             else:
#                 signup(json_file_path)

#     elif page == "📊 Dashboard":
#         if st.session_state.get("logged_in"):
#             st.markdown("<h2 class='gradient-header'>📊 Your Wellness Dashboard</h2>", unsafe_allow_html=True)
#             render_dashboard(st.session_state["user_info"])
#         else:
#             st.warning("🔒 Please login or sign up to access your dashboard.")
#             login_or_signup = st.radio(
#                 "Choose an option:",
#                 ("🔑 Login", "✨ Signup"),
#                 key="login_signup_dash",
#                 horizontal=True
#             )
#             if login_or_signup == "🔑 Login":
#                 login(json_file_path)
#             else:
#                 signup(json_file_path)

#     elif page == "🔍 Mental Health Assessment":
#         if st.session_state.get("logged_in"):
#             st.markdown("<h2 class='gradient-header'>🔍 Mental Health Assessment</h2>", unsafe_allow_html=True)
            
#             # Assessment introduction
#             st.markdown("""
#             <div class="card">
#                 <h3>How It Works</h3>
#                 <p>This assessment analyzes your voice patterns and speech content to provide insights about your mental wellbeing.</p>
#                 <ol>
#                     <li>Record or upload a voice sample where you describe how you've been feeling lately</li>
#                     <li>Our AI analyzes both your voice patterns and the content of your speech</li>
#                     <li>You'll receive personalized recommendations based on the assessment</li>
#                 </ol>
#                 <p><em>Note: This is not a clinical diagnosis. For medical advice, please consult a healthcare professional.</em></p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Reset assessment button
#             if st.session_state.analysis_complete:
#                 if st.button("🔄 Start New Assessment", key="reset_assessment"):
#                     st.session_state.voice_result = None
#                     st.session_state.voice_prob = 0.0
#                     st.session_state.text_result = None
#                     st.session_state.audio_data = None
#                     st.session_state.analysis_complete = False
#                     st.session_state.show_results = False
#                     st.session_state.current_diagnosis = None
#                     st.session_state.current_status = None
#                     st.rerun()
            
#             # Check if models are loaded
#             if not voice_model_loaded or not text_model_loaded:
#                 st.warning("⚠️ Some analysis models could not be loaded. Assessment functionality may be limited.")
            
#             # Only show the voice input if we're not already showing results
#             if not st.session_state.show_results:
#                 # Voice Analysis Section
#                 st.markdown("<h3 style='margin-top: 30px;'>🎙️ Voice Input</h3>", unsafe_allow_html=True)
                
#                 options = ["🎤 Record", "📁 Upload"]
#                 choice = st.radio("Choose an option for voice input:", options, horizontal=True)
                
#                 st.markdown("""
#                 <div style="background-color: #f1f5f9; padding: 15px; border-radius: 5px; margin: 15px 0;">
#                     <p><strong>Suggestion:</strong> For the best results, speak for at least 30 seconds about how you've been feeling 
#                     lately, your mood, energy levels, sleep patterns, and general mental state.</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 if choice == "🎤 Record":
#                     st.write("Click the button below to start recording:")
#                     audio = st_audiorec()
#                     if audio is not None:
#                         st.session_state.audio_data = audio
#                         st.success("✅ Recording saved! Click 'Analyze Recording' to continue.")
                        
#                         if st.button('🔍 Analyze Recording', key="analyze_recording"):
#                             # Show a processing animation
#                             with st.spinner("🔄 Processing your assessment..."):
#                                 # Save audio for analysis
#                                 with open("temp_analysis.wav", "wb") as f:
#                                     f.write(audio)
                                
#                                 # Create a progress bar
#                                 progress_bar = st.progress(0)
                                
#                                 # Analyze voice for depression
#                                 progress_bar.progress(25)
#                                 time.sleep(0.5)  # Simulate processing time
#                                 st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                                
#                                 progress_bar.progress(50)
#                                 time.sleep(0.5)  # Simulate processing time
                                
#                                 # Initialize text result as normal (fallback)
#                                 st.session_state.text_result = "normal"
                                
#                                 # Only attempt text analysis if the model was loaded properly
#                                 if text_model_loaded:
#                                     try:
#                                         # Get transcribed text
#                                         transcribed_text = transcribe_audio_from_data(audio)
                                        
#                                         progress_bar.progress(75)
#                                         time.sleep(0.5)  # Simulate processing time
                                        
#                                         if transcribed_text:
#                                             # Process text using the TF-IDF vectorizer
#                                             processed_text = preprocess(transcribed_text)
                                            
#                                             # Use the vectorizer to transform the text
#                                             tfidf_vector = vectorizer.transform([processed_text])
                                            
#                                             # Get prediction from text model
#                                             text_pred = text_model.predict(tfidf_vector)
#                                             st.session_state.text_result = text_pred[0]
#                                     except Exception as e:
#                                         # If there's any error in text analysis, silently fallback
#                                         pass
                                
#                                 # Complete the progress bar
#                                 progress_bar.progress(100)
                                
#                                 # Get the combined diagnosis
#                                 st.session_state.current_diagnosis, st.session_state.current_status = get_combined_diagnosis(
#                                     st.session_state.voice_result, 
#                                     st.session_state.text_result
#                                 )
                                
#                                 st.session_state.analysis_complete = True
#                                 st.session_state.show_results = True
                                
#                                 # Remove temporary file
#                                 if os.path.exists("temp_analysis.wav"):
#                                     os.remove("temp_analysis.wav")
                            
#                             # Refresh the page to show results
#                             st.rerun()
                
#                 elif choice == "📁 Upload":
#                     audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
#                     if audio is not None:
#                         st.audio(audio, format="audio/wav")
#                         audio_bytes = audio.read()
#                         st.session_state.audio_data = audio_bytes
#                         st.success("✅ Audio uploaded! Click 'Analyze Upload' to continue.")
                        
#                         if st.button('🔍 Analyze Upload', key="analyze_upload"):
#                             # Show a processing animation
#                             with st.spinner("🔄 Processing your assessment..."):
#                                 # Save audio for analysis
#                                 with open("temp_analysis.wav", "wb") as f:
#                                     f.write(audio_bytes)
                                
#                                 # Create a progress bar
#                                 progress_bar = st.progress(0)
                                
#                                 # Analyze voice for depression
#                                 progress_bar.progress(25)
#                                 time.sleep(0.5)  # Simulate processing time
#                                 st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                                
#                                 progress_bar.progress(50)
#                                 time.sleep(0.5)  # Simulate processing time
                                
#                                 # Initialize text result as normal (fallback)
#                                 st.session_state.text_result = "normal"
                                
#                                 # Only attempt text analysis if the model was loaded properly
#                                 if text_model_loaded:
#                                     try:
#                                         # Get transcribed text
#                                         transcribed_text = transcribe_audio_from_data(audio_bytes)
                                        
#                                         progress_bar.progress(75)
#                                         time.sleep(0.5)  # Simulate processing time
                                        
#                                         if transcribed_text:
#                                             # Process text using the TF-IDF vectorizer
#                                             processed_text = preprocess(transcribed_text)
                                            
#                                             # Use the vectorizer to transform the text
#                                             tfidf_vector = vectorizer.transform([processed_text])
                                            
#                                             # Get prediction from text model
#                                             text_pred = text_model.predict(tfidf_vector)
#                                             st.session_state.text_result = text_pred[0]
#                                     except Exception as e:
#                                         # If there's any error in text analysis, silently fallback
#                                         pass
                                
#                                 # Complete the progress bar
#                                 progress_bar.progress(100)
                                
#                                 # Get the combined diagnosis
#                                 st.session_state.current_diagnosis, st.session_state.current_status = get_combined_diagnosis(
#                                     st.session_state.voice_result, 
#                                     st.session_state.text_result
#                                 )
                                
#                                 st.session_state.analysis_complete = True
#                                 st.session_state.show_results = True
                                
#                                 # Remove temporary file
#                                 if os.path.exists("temp_analysis.wav"):
#                                     os.remove("temp_analysis.wav")
                            
#                             # Refresh the page to show results
#                             st.rerun()
            
#             # Display final combined results if analysis is complete and showing results
#             if st.session_state.analysis_complete and st.session_state.show_results:
#                 display_results_and_recommendations(
#                     st.session_state.current_diagnosis,
#                     st.session_state.current_status
#                 )
#         else:
#             st.warning("🔒 Please login or signup to use the mental health assessment.")
#             login_or_signup = st.radio(
#                 "Choose an option:",
#                 ("🔑 Login", "✨ Signup"),
#                 key="login_signup_assess",
#                 horizontal=True
#             )
#             if login_or_signup == "🔑 Login":
#                 login(json_file_path)
#             else:
#                 signup(json_file_path)
            
# if __name__ == "__main__":
#     initialize_database()
#     main()



import streamlit as st
import json
import os
import re
import string
from st_audiorec import st_audiorec
import io
import whisper
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from pydub import AudioSegment
import pickle
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
from PIL import Image
from io import BytesIO
import time
import random
from datetime import datetime
from emotion_integration import analyze_emotion,analyze_text_emotion

# Set page configuration
st.set_page_config(
    page_title="MindWell - Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to improve aesthetics
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #6a0dad;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #4a0a8c;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .logout-btn {
            background-color: #ff5252 !important;
        }
        .logout-btn:hover {
            background-color: #ff1a1a !important;
        }
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
            font-family: 'Helvetica Neue', Helvetica, sans-serif;
        }
        .resource-card {
            background-color: #f1f5f9;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #6a0dad;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .css-1d391kg {
            padding-top: 3.5rem;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#f8f9fa,#e9ecef);
        }
        /* Custom card styling */
        .card {
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
        }
        /* Status colors */
        .status-danger {
            border-left: 5px solid #ff4b4b !important;
            background-color: rgba(255, 75, 75, 0.05);
        }
        .status-warning {
            border-left: 5px solid #ffa62b !important;
            background-color: rgba(255, 166, 43, 0.05);
        }
        .status-success {
            border-left: 5px solid #0ead69 !important;
            background-color: rgba(14, 173, 105, 0.05);
        }
        .status-info {
            border-left: 5px solid #4361ee !important;
            background-color: rgba(67, 97, 238, 0.05);
        }
        .resource-title {
            font-weight: bold;
            color: #6a0dad;
        }
        /* Recommendation Section */
        .recommendation-section {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
        .recommendation-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 4px solid #6a0dad;
        }
        .recommendation-title {
            font-weight: bold;
            color: #6a0dad;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        /* Emoji styling */
        .emoji {
            font-size: 1.5em;
            margin-right: 0.5em;
            display: inline-block;
        }
        /* Section headers with gradients */
        .gradient-header {
            background: linear-gradient(90deg, #6a0dad, #9b59b6);
            color: white !important;
            padding: 10px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        /* Animation for assessment in progress */
        @keyframes pulse {
            0% {
                transform: scale(1);
                opacity: 1;
            }
            50% {
                transform: scale(1.05);
                opacity: 0.8;
            }
            100% {
                transform: scale(1);
                opacity: 1;
            }
        }
        .pulse-animation {
            animation: pulse 1.5s infinite;
        }
        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: #6a0dad;
        }
        /* Hide hamburger menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Apply CSS
local_css()

# Function to create a custom header with emoji
def custom_header(text, emoji, level=2):
    emoji_span = f'<span class="emoji">{emoji}</span>'
    if level == 1:
        st.markdown(f'<h1>{emoji_span} {text}</h1>', unsafe_allow_html=True)
    elif level == 2:
        st.markdown(f'<h2>{emoji_span} {text}</h2>', unsafe_allow_html=True)
    elif level == 3:
        st.markdown(f'<h3>{emoji_span} {text}</h3>', unsafe_allow_html=True)

# Function to load and display the logo
def display_logo():
    # Create a simple logo with PIL if no logo file exists
    img = Image.new('RGB', (200, 200), color = (106, 13, 173))
    
    # Add some basic text to the image
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Try to create a circular shape
    draw.ellipse((10, 10, 190, 190), fill=(255, 255, 255))
    draw.ellipse((20, 20, 180, 180), fill=(106, 13, 173))
    
    # Try to add the letter M
    try:
        # This will work if a suitable font is available
        font = ImageFont.truetype("arial.ttf", 120)
        draw.text((60, 40), "M", fill=(255, 255, 255), font=font)
    except:
        # Fallback if font is not available
        try:
            font = ImageFont.load_default()
            draw.text((80, 80), "M", fill=(255, 255, 255), font=font)
        except:
            pass
    
    d = BytesIO()
    img.save(d, 'PNG')
    
    # Create a colored background with the app name
    st.markdown(f"""
    <div style="display: flex; align-items: center; background: linear-gradient(90deg, #6a0dad, #9b59b6); padding: 1rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <img src="data:image/png;base64,{base64.b64encode(d.getvalue()).decode()}" style="width: 80px; height: 80px; border-radius: 50%; margin-right: 20px; background-color: white; padding: 5px;">
        <div>
            <h1 style="margin: 0; color: white; font-family: 'Helvetica Neue', Helvetica, sans-serif;">MindWell 🧠</h1>
            <p style="margin: 0; color: rgba(255,255,255,0.8);">Mental Health Assessment Platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = None
if "voice_result" not in st.session_state:
    st.session_state.voice_result = None
    st.session_state.voice_prob = 0.0
    st.session_state.text_result = None
    st.session_state.audio_data = None
    st.session_state.analysis_complete = False
    st.session_state.show_results = False
    st.session_state.current_diagnosis = None
    st.session_state.current_status = None
    st.session_state.input_method = None
    st.session_state.text_input_data = None
if "emotion_result" not in st.session_state:
    st.session_state.emotion_result = None
# Load text classification model
try:
    with open('logistic.pkl', 'rb') as f:
        text_model = pickle.load(f)

    with open('tfidf_vectorizer.pik', 'rb') as f:
        vectorizer = pickle.load(f)
    text_model_loaded = True
except Exception as e:
    text_model_loaded = False
    
# Load voice depression detection model (MLP)
try:
    voice_model = load_model("mlp_model.keras")
    voice_model_loaded = True
except Exception as e:
    voice_model_loaded = False

# Helper functions for voice feature extraction
def noise(data):
    """Add random noise to the audio data"""
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    """Stretch the audio data"""
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sampling_rate, pitch_factor=0.7):
    """Change the pitch of the audio data"""
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data, sample_rate):
    """Extract audio features for depression detection"""
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    # Spectral Contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_contrast))

    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonnetz))

    # Rolloff
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, rolloff))

    # Zero Crossing Rate Variance
    zcr_var = np.var(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr_var))

    return result

def analyze_voice(audio_path, sample_rate=22050):
    """Analyze voice for depression detection using MLP model"""
    if not voice_model_loaded:
        return "model_not_loaded", 0.0
    
    try:
        # Extract features from the audio
        data, _ = librosa.load(audio_path, sr=sample_rate, duration=2.5, offset=0.6)
        features = extract_features(data, sample_rate)
        
        # Reshape features for model input
        features = np.array([features])
        
        # Make prediction using the MLP model
        prediction = voice_model.predict(features)
        
        # Get class with highest probability
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Map prediction to label (0 is Depressed, 1 is Non-Depressed)
        if predicted_class == 0:
            return "depressed", confidence
        else:
            return "normal", confidence
    except Exception as e:
        return "error", 0.0

def analyze_text(text_input):
    """Analyze text for depression or suicidal thoughts"""
    if not text_model_loaded:
        return "model_not_loaded"
    
    try:
        # Process text using the TF-IDF vectorizer
        processed_text = preprocess(text_input)
        
        # Use the vectorizer to transform the text
        tfidf_vector = vectorizer.transform([processed_text])
        
        # Get prediction from text model
        text_pred = text_model.predict(tfidf_vector)
        return text_pred[0]
    except Exception as e:
        return "error"

def logout():
    """Function to handle user logout"""
    for key in list(st.session_state.keys()):
        if key != "user_index":  
            del st.session_state[key]
    st.session_state["logged_in"] = False
    st.session_state["user_info"] = None
    st.session_state.voice_result = None
    st.session_state.voice_prob = 0.0
    st.session_state.text_result = None
    st.session_state.audio_data = None
    st.session_state.analysis_complete = False
    st.session_state.show_results = False
    st.session_state.input_method = None
    st.session_state.text_input_data = None
    st.rerun()

def signup(json_file_path="data.json"):
    """Handle user signup"""
    st.markdown("<h2 class='gradient-header'>✨ Create Your Account</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("signup_form", border=False):
            st.markdown("<p style='text-align: center;'>Fill in the details below to start your mental wellness journey:</p>", unsafe_allow_html=True)
            name = st.text_input("🙋 Full Name")
            email = st.text_input("📧 Email Address")
            
            col_age, col_sex = st.columns(2)
            with col_age:
                age = st.number_input("🎂 Age", min_value=18, max_value=120, value=25)
            with col_sex:
                sex = st.selectbox("⚧ Gender", ("Male", "Female", "Other"))
            
            password = st.text_input("🔒 Password", type="password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password")

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("✅ Create Account", use_container_width=True)
            st.markdown("""
            <p style='text-align: center; font-size: 0.8em; margin-top: 10px;'>
                By creating an account, you agree to our Terms of Service and Privacy Policy.
            </p>
            """, unsafe_allow_html=True)

            if submitted:
                if password == confirm_password:
                    user = create_account(name, email, age, sex, password, json_file_path)
                    st.session_state["logged_in"] = True
                    st.session_state["user_info"] = user
                    st.success("🎉 Account created successfully! Welcome to MindWell.")
                    st.rerun()
                else:
                    st.error("❌ Passwords do not match. Please try again.")

def check_login(username, password, json_file_path="data.json"):
    """Verify user login credentials"""
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["user_info"] = user
                return user

        return None
    except Exception as e:
        return None

def initialize_database(json_file_path="data.json"):
    """Initialize the JSON database if it doesn't exist"""
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")
        
def create_account(name, email, age, sex, password, json_file_path="data.json"):
    """Create a new user account"""
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "assessment_history": []
        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def login(json_file_path="data.json"):
    """Handle user login"""
    st.markdown("<h2 class='gradient-header'>👋 Welcome Back</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form", border=False):
            st.markdown("<p style='text-align: center;'>Sign in to continue your mental wellness journey</p>", unsafe_allow_html=True)
            username = st.text_input("📧 Email Address")
            password = st.text_input("🔒 Password", type="password")

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔑 Login", use_container_width=True)

            if submitted:
                user = check_login(username, password, json_file_path)
                if user is not None:
                    st.success("✅ Login successful! Welcome back.")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    """Get user information from the database"""
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        return None

def save_assessment_result(user_email, result, status, json_file_path="data.json"):
    """Save assessment result to user history"""
    try:
        # Read the current data
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
        
        # Find the user
        for user in data["users"]:
            if user["email"] == user_email:
                # Initialize assessment_history if it doesn't exist
                if "assessment_history" not in user:
                    user["assessment_history"] = []
                
                # Add the new assessment result with timestamp
                assessment_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "result": result,
                    "status": status
                }
                
                # Only keep the most recent 5 entries
                user["assessment_history"] = [assessment_entry] + user["assessment_history"][:4]
                break
        
        # Save the updated data
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
            
    except Exception as e:
        print(f"Error saving assessment result: {e}")

def update_user_info():
    """Update user info in session state with latest from database"""
    if st.session_state.get("logged_in") and st.session_state.get("user_info"):
        user_email = st.session_state["user_info"]["email"]
        updated_info = get_user_info(user_email)
        if updated_info:
            st.session_state["user_info"] = updated_info

def render_dashboard(user_info, json_file_path="data.json"):
    """Render user dashboard"""
    update_user_info()  # Make sure we have latest user data
    
    # User welcome card
    st.markdown(f"""
    <div class="card">
        <h2 style="color: #6a0dad; margin-bottom: 20px;">👋 Welcome, {user_info['name']}!</h2>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 250px;">
                <h3>👤 Personal Information</h3>
                <p><strong>Name:</strong> {user_info['name']}</p>
                <p><strong>Email:</strong> {user_info['email']}</p>
                <p><strong>Age:</strong> {user_info['age']}</p>
                <p><strong>Gender:</strong> {user_info['sex']}</p>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <h3>💡 Mental Wellness Tips</h3>
                <ul>
                    <li>🧘 Practice mindfulness for 10 minutes daily</li>
                    <li>☕ Take short breaks when feeling overwhelmed</li>
                    <li>👥 Connect with loved ones regularly</li>
                    <li>😴 Maintain a consistent sleep schedule</li>
                    <li>🚶 Stay physically active with activities you enjoy</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Assessment history (if any)
    if "assessment_history" in user_info and user_info["assessment_history"]:
        st.markdown("""
        <div class="card">
            <h3 style="color: #6a0dad; margin-bottom: 15px;">📊 Your Assessment History</h3>
            <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="border-bottom: 2px solid #ddd; padding: 8px; text-align: left;">Date</th>
                        <th style="border-bottom: 2px solid #ddd; padding: 8px; text-align: left;">Result</th>
                    </tr>
                </thead>
                <tbody>
        """, unsafe_allow_html=True)
        
        for entry in user_info["assessment_history"]:
            # Define status color based on assessment result
            if entry["status"] == "danger":
                status_color = "#ff4b4b"
            elif entry["status"] == "warning":
                status_color = "#ffa62b"
            elif entry["status"] == "success":
                status_color = "#0ead69"
            else:
                status_color = "#4361ee"
                
            st.markdown(f"""
            <tr>
                <td style="border-bottom: 1px solid #ddd; padding: 8px; text-align: left;">{entry["date"]}</td>
                <td style="border-bottom: 1px solid #ddd; padding: 8px; text-align: left;">
                    <span style="color: {status_color}; font-weight: bold;">{entry["result"]}</span>
                </td>
            </tr>
            """, unsafe_allow_html=True)
            
        st.markdown("""
                </tbody>
            </table>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Mental wellness resources
    st.markdown("""
    <div class="card">
        <h3 style="color: #6a0dad;">🌟 Mental Wellness Resources</h3>
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px;">
            <div class="resource-card" style="flex: 1; min-width: 200px;">
                <h4 class="resource-title">📱 Meditation Apps</h4>
                <ul>
                    <li>Headspace</li>
                    <li>Calm</li>
                    <li>Insight Timer</li>
                </ul>
            </div>
            <div class="resource-card" style="flex: 1; min-width: 200px;">
                <h4 class="resource-title">🌱 Self-Care Activities</h4>
                <ul>
                    <li>Journaling daily thoughts</li>
                    <li>Nature walks and fresh air</li>
                    <li>Creative expression through art</li>
                </ul>
            </div>
            <div class="resource-card" style="flex: 1; min-width: 200px;">
                <h4 class="resource-title">👥 Support Resources</h4>
                <ul>
                    <li>Local support groups</li>
                    <li>Online communities for connection</li>
                    <li>Professional therapy options</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Inspirational quote card - randomly selected
    quotes = [
        {"text": "You don't have to be positive all the time. It's perfectly okay to feel sad, angry, annoyed, frustrated, scared and anxious. Having feelings doesn't make you a negative person. It makes you human.", "author": "Lori Deschene"},
        {"text": "Mental health problems don't define who you are. They are something you experience, but they are not you.", "author": "Unknown"},
        {"text": "You are not alone in this. You have more support than you could ever know.", "author": "Unknown"},
        {"text": "Your mental health is a priority. Your happiness is essential. Your self-care is a necessity.", "author": "Unknown"},
        {"text": "Recovery is not one and done. It is a lifelong journey that takes place one day, one step at a time.", "author": "Unknown"}
    ]
    
    random_quote = random.choice(quotes)
    
    st.markdown(f"""
    <div class="card" style="background-color: #f0e6ff; border-left: 5px solid #6a0dad;">
        <div style="font-style: italic; font-size: 1.1em; margin-bottom: 10px;">"{random_quote["text"]}"</div>
        <div style="text-align: right; font-weight: bold;">— {random_quote["author"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
def preprocess(text): 
    """Preprocess text for depression analysis"""
    def remove_url(text):
        return re.sub(r"http\S+", "", text) 
    exclude = string.punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", exclude))
    
    def remove_stopwords(text):
        stopword = stopwords.words('english')
        new_text = []
        for word in text.split():
            if word in stopword:
                new_text.append('')
            else:
                new_text.append(word)
        x = new_text[:]
        new_text.clear()
        return " ".join(x)
    def lemmatize_text(text):
        words = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)
    text = remove_url(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text
   
def transcribe_audio_from_data(file_data):
    """Transcribe audio to text"""
    try:
        with open("temp.mp3", "wb") as f:
            f.write(file_data)
        model = whisper.load_model("base")
        result = model.transcribe("temp.mp3")
        os.remove("temp.mp3")
        return result['text']
    except Exception as e:
        return ""

def get_combined_diagnosis(voice_result, text_result, emotion_result=None, input_method=None):
    """Combine voice, text, and emotion analysis results according to the specified rules"""
    
    # For text-only input, we don't have voice analysis
    if input_method == "text_only":
        diagnosis, status = "", ""
        if text_result == "depression":
            diagnosis = "Depression detected"
            status = "warning"
        elif text_result == "suicide":
            diagnosis = "Suicide risk detected"
            status = "danger"
        elif text_result == "normal" or text_result == "non-suicide":
            diagnosis = "Non-suicidal"
            status = "success"
        else:
            diagnosis = "Assessment inconclusive"
            status = "info"
        
        # Add emotion to diagnosis if available
        if emotion_result and emotion_result != "Unknown":
            diagnosis += f" with {emotion_result} emotion"
        
        return diagnosis, status

    # If voice model had errors
    if voice_result == "error" or voice_result == "model_not_loaded":
        diagnosis = "Voice analysis unavailable, assessment incomplete"
        status = "info"
        # Add emotion to diagnosis if available
        if emotion_result and emotion_result != "Unknown":
            diagnosis += f" with {emotion_result} emotion"
        return diagnosis, status
    
    # Normal combined voice and text analysis
    text_normal = text_result == "normal" or text_result == "non-suicide"
    
    if voice_result == "depressed" and text_result == "depression":
        diagnosis = "Depression detected"
        status = "warning"
    elif voice_result == "depressed" and text_result == "suicide":
        diagnosis = "Suicide risk detected" 
        status = "danger"
    elif voice_result == "depressed" and text_normal:
        diagnosis = "Non-suicidal"
        status = "info"
    elif voice_result == "normal" and text_normal:
        diagnosis = "Non-suicidal"
        status = "success"
    elif voice_result == "normal" and text_result == "suicide":
        diagnosis = "Suicide risk detected"
        status = "danger"
    elif voice_result == "normal" and text_result == "depression":
        diagnosis = "Depression detected"
        status = "warning"
    else:
        diagnosis = "Assessment inconclusive"
        status = "info"
    
    # Add emotion to diagnosis if available
    if emotion_result and emotion_result != "Unknown":
        diagnosis += f" with {emotion_result} emotion"
    
    return diagnosis, status

def get_recommendations(diagnosis):
    """Get recommendations based on the diagnosis"""
    
    # Common recommendations for all mental health conditions
    common_recs = [
        "Practice mindfulness meditation daily",
        "Maintain a regular sleep schedule",
        "Stay physically active with activities you enjoy",
        "Connect with supportive friends and family",
        "Limit consumption of news and social media"
    ]
    
    # Specific recommendations based on diagnosis
    if "Suicide risk" in diagnosis:
        specific_recs = [
            "Contact a crisis helpline immediately (988 or 1-800-273-8255)",
            "Go to your nearest emergency room if thoughts are severe",
            "Remove access to potential means of self-harm",
            "Don't be alone - stay with someone you trust",
            "Work with a professional to create a safety plan"
        ]
        emoji = "🚨"
        color = "#ff4b4b"
    elif "Depression" in diagnosis:
        specific_recs = [
            "Schedule an appointment with a mental health professional",
            "Consider starting a mood journal to track patterns",
            "Set small, achievable daily goals",
            "Expose yourself to natural sunlight daily",
            "Practice gratitude by noting three positive things each day"
        ]
        emoji = "💜"
        color = "#ffa62b"
    elif "Non-suicidal" in diagnosis:
        specific_recs = [
            "Continue building healthy mental habits",
            "Explore new activities that bring you joy",
            "Learn stress management techniques",
            "Consider regular check-ins with a mental health professional",
            "Share your wellness journey with others"
        ]
        emoji = "✨"
        color = "#0ead69"
    else:  # Assessment inconclusive
        specific_recs = [
            "Consider a more comprehensive assessment with a professional",
            "Keep a mood journal to track your emotions",
            "Learn more about different mental health conditions",
            "Try the assessment again when you're feeling different",
            "Focus on general wellness practices"
        ]
        emoji = "🔍"
        color = "#4361ee"
    
    return {
        "common": common_recs,
        "specific": specific_recs,
        "emoji": emoji,
        "color": color
    }

def display_results_and_recommendations(diagnosis, status):
    if status == "danger":
        status_class = "status-danger"
        emoji = "🚨"
    elif status == "warning":
        status_class = "status-warning"
        emoji = "⚠️"
    elif status == "success":
        status_class = "status-success"
        emoji = "✅"
    else:
        status_class = "status-info"
        emoji = "ℹ️"
        
    # Create emotion display HTML if available
    emotion_html = ""
    if st.session_state.get("emotion_result") and st.session_state.emotion_result != "Unknown":
        # Make sure emotion name is capitalized for consistent display
        emotion_name = st.session_state.emotion_result.capitalize()
        
        emotion_emoji = {
            "Happy": "😊", 
            "Sad": "😢", 
            "Angry": "😠", 
            "Fear": "😨", 
            "Surprise": "😲", 
            "Disgust": "🤢", 
            "Neutral": "😐",
            "Love": "❤️"
        }.get(emotion_name, "🙂")
        
        emotion_html = f"""
        {emotion_emoji} {emotion_name} 
           
        """
    
    # Display the result card (only once)
    st.markdown(f"""
    <div class="card {status_class}">
        <h2 style="display:flex;align-items:center;gap:10px;">{emoji} Assessment Result</h2>
        <p style="font-size: 1.5em; font-weight: bold; margin: 20px 0;">{diagnosis}</p>
        <p style="font-style: italic;">Assessment completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        {emotion_html}
    </div>
    """, unsafe_allow_html=True)

    # Get recommendations
    recs = get_recommendations(diagnosis)

    # Display recommendation section
    st.markdown(f"""
    <div class="card">
        <h3 style="display:flex;align-items:center;gap:10px;">{recs['emoji']} Personalized Recommendations</h3>
        <p>Based on your assessment, we recommend the following:</p>
        
    
          
    """, unsafe_allow_html=True)

    # Display specific recommendations
    for rec in recs['specific']:
        st.markdown(f"""
        <div style="background-color: {recs['color']}15; border-left: 3px solid {recs['color']}; 
                    padding: 15px; border-radius: 5px; flex: 1; min-width: 250px;">
            <p style="margin: 0;">{rec}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
            </div>
        </div>
        
        <div style="margin-top: 20px;">
            <h3>General Wellness Practices</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
    """, unsafe_allow_html=True)

    # Display common recommendations
    for rec in recs['common']:
        st.markdown(f"""
        <div style="background-color: #f1f5f9; border-left: 3px solid #6a0dad; 
                    padding: 15px; border-radius: 5px; flex: 1; min-width: 250px;">
            <p style="margin: 0;">{rec}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
#     st.markdown("""
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

    # Resources section
    st.markdown("""
    <div class="card">
        <h2 style="display:flex;align-items:center;gap:10px;">📚 Helpful Resources</h2>
    """, unsafe_allow_html=True)

    # Resources based on diagnosis
    if "Suicide risk" in diagnosis:
        st.markdown("""
        <div style="margin-top: 15px;">
            <h3>Crisis Resources</h3>
            <ul>
                <li><strong><a href="https://988lifeline.org/" target="_blank">988 Suicide & Crisis Lifeline</a></strong> - Call or text 988</li>
                <li><strong><a href="https://www.crisistextline.org/" target="_blank">Crisis Text Line</a></strong> - Text HOME to 741741</li>
                <li><strong><a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank">International Association for Suicide Prevention</a></strong> - Global resources</li>
                <li><strong><a href="https://www.betterhelp.com/" target="_blank">BetterHelp</a></strong> - Online counseling</li>
                <li><strong><a href="https://suicidepreventionlifeline.org/wp-content/uploads/2016/08/Brown_StanleySafetyPlanTemplate.pdf" target="_blank">Safety Plan Template</a></strong> - Downloadable safety plan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif "Depression" in diagnosis:
        st.markdown("""
        <div style="margin-top: 15px;">
            <h3>Depression Resources</h3>
            <ul>
                <li><strong><a href="https://www.dbsalliance.org/" target="_blank">Depression and Bipolar Support Alliance</a></strong> - Support groups and resources</li>
                <li><strong><a href="https://www.nimh.nih.gov/health/topics/depression" target="_blank">National Institute of Mental Health</a></strong> - Information about depression</li>
                <li><strong><a href="https://www.psychologytoday.com/us/therapists" target="_blank">Psychology Today Therapist Finder</a></strong> - Find a therapist near you</li>
                <li><strong><a href="https://www.betterhelp.com/" target="_blank">BetterHelp</a></strong> - Online counseling</li>
                <li><strong><a href="https://www.headspace.com/" target="_blank">Headspace</a></strong> - Meditation app</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top: 15px;">
            <h3>Mental Wellness Resources</h3>
            <ul>
                <li><strong><a href="https://www.nami.org/" target="_blank">National Alliance on Mental Illness</a></strong> - Mental health education and support</li>
                <li><strong><a href="https://www.mhanational.org/" target="_blank">Mental Health America</a></strong> - Mental health resources</li>
                <li><strong><a href="https://www.headspace.com/" target="_blank">Headspace</a></strong> - Meditation app</li>
                <li><strong><a href="https://www.calm.com/" target="_blank">Calm</a></strong> - Sleep and meditation app</li>
                <li><strong><a href="https://www.betterhelp.com/" target="_blank">BetterHelp</a></strong> - Online counseling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; font-size: 0.9em;">
        <p><strong>Disclaimer:</strong> This assessment is not a clinical diagnosis. If you're experiencing severe symptoms, 
        please consult with a qualified mental health professional or go to your nearest emergency room.</p>
    </div>
    """, unsafe_allow_html=True)

    # Save this assessment to user history
    if st.session_state.get("logged_in") and st.session_state.get("user_info"):
        save_assessment_result(
            st.session_state["user_info"]["email"], 
            diagnosis,
            status
        )
# Replace the display_results_and_recommendations function with this improved version

# def display_results_and_recommendations(diagnosis, status):
#     """Display assessment results and recommendations"""
#     # Set status-specific styling
#     if status == "danger":
#         status_class = "status-danger"
#         emoji = "🚨"
#     elif status == "warning":
#         status_class = "status-warning"
#         emoji = "⚠️"
#     elif status == "success":
#         status_class = "status-success"
#         emoji = "✅"
#     else:
#         status_class = "status-info"
#         emoji = "ℹ️"
        
#     # Create emotion display HTML if available
#     emotion_html = ""
#     if st.session_state.get("emotion_result") and st.session_state.emotion_result != "Unknown":
#         # Make sure emotion name is capitalized for consistent display
#         emotion_name = st.session_state.emotion_result.capitalize()
        
#         emotion_emoji = {
#             "Happy": "😊", 
#             "Sad": "😢", 
#             "Angry": "😠", 
#             "Fear": "😨", 
#             "Surprise": "😲", 
#             "Disgust": "🤢", 
#             "Neutral": "😐",
#             "Love": "❤️"
#         }.get(emotion_name, "🙂")
        
#         emotion_html = f"""
#         <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-top: 15px;">
#             <h3>Detected Emotion</h3>
#             <p style="font-size: 1.8em; display: flex; align-items: center;">
#                 <span style="font-size: 2em; margin-right: 15px;">{emotion_emoji}</span>
#                 <strong>{emotion_name}</strong>
#             </p>
#         </div>
#         """
    
#     # Display the result card (only once)
#     st.markdown(f"""
#     <div class="card {status_class}">
#         <h2 style="display:flex;align-items:center;gap:10px;">{emoji} Assessment Result</h2>
#         <p style="font-size: 1.5em; font-weight: bold; margin: 20px 0;">{diagnosis}</p>
#         <p style="font-style: italic;">Assessment completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
#         {emotion_html}
#     </div>
#     """, unsafe_allow_html=True)

#     # Get recommendations
#     recs = get_recommendations(diagnosis)

#     # Display recommendation section
#     st.markdown(f"""
#     <div class="card">
#         <h3 style="display:flex;align-items:center;gap:10px;">{recs['emoji']} Personalized Recommendations</h3>
#         <p>Based on your assessment, we recommend the following:</p>
        
#         <div style="margin-top: 15px;">
#             <h4>Priority Actions</h4>
#             <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
#     """, unsafe_allow_html=True)

#     # Display specific recommendations
#     for rec in recs['specific']:
#         st.markdown(f"""
#         <div style="background-color: {recs['color']}15; border-left: 3px solid {recs['color']}; 
#                     padding: 15px; border-radius: 5px; flex: 1; min-width: 250px;">
#             <p style="margin: 0;">{rec}</p>
#         </div>
#         """, unsafe_allow_html=True)

#     st.markdown("""
#             </div>
#         </div>
        
#         <div style="margin-top: 20px;">
#             <h3>General Wellness Practices</h3>
#             <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
#     """, unsafe_allow_html=True)

#     # Display common recommendations
#     for rec in recs['common']:
#         st.markdown(f"""
#         <div style="background-color: #f1f5f9; border-left: 3px solid #6a0dad; 
#                     padding: 15px; border-radius: 5px; flex: 1; min-width: 250px;">
#             <p style="margin: 0;">{rec}</p>
#         </div>
#         """, unsafe_allow_html=True)

#     st.markdown("""
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

    # Rest of the function remains the same...
def main(json_file_path="data.json"):
    # Display logo
    display_logo()
    
    # Create sidebar
    st.sidebar.markdown("<h2 style='text-align: center;'>🧠 MindWell</h2>", unsafe_allow_html=True)
    
    # Add logout button if logged in
    if st.session_state.get("logged_in"):
        st.sidebar.button("📤 Logout", on_click=logout, key="logout_button", type="primary", 
                         help="Click to log out of your account")
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    
    # Navigation menu
    page = st.sidebar.radio(
        "📋 Navigation",
        ("👤 Account", "📊 Dashboard", "🔍 Mental Health Assessment"),
        key="navigation",
    )
    
    # Add sidebar footer
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px; background-color: #f1f5f9; border-radius: 10px;'>
        <p style='font-size: 0.8em; margin-bottom: 5px;'>MindWell Health Assessment</p>
        <p style='font-size: 0.7em; color: #666;'>Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)

    # Page content
    if page == "👤 Account":
        if st.session_state.get("logged_in"):
            st.markdown("<h2 class='gradient-header'>👤 Account Settings</h2>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
                <h3>Your Profile</h3>
                <p>You are currently logged in. Use the navigation menu to access different features.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Logout button on the page as well
            st.button("📤 Logout from MindWell", on_click=logout, key="logout_button_page", 
                     type="primary", help="Click to log out of your account")
        else:
            st.title("Welcome to MindWell")
            login_or_signup = st.radio(
                "Choose an option:",
                ("🔑 Login", "✨ Signup"),
                key="login_signup",
                horizontal=True
            )
            if login_or_signup == "🔑 Login":
                login(json_file_path)
            else:
                signup(json_file_path)

    elif page == "📊 Dashboard":
        if st.session_state.get("logged_in"):
            st.markdown("<h2 class='gradient-header'>📊 Your Wellness Dashboard</h2>", unsafe_allow_html=True)
            render_dashboard(st.session_state["user_info"])
        else:
            st.warning("🔒 Please login or sign up to access your dashboard.")
            login_or_signup = st.radio(
                "Choose an option:",
                ("🔑 Login", "✨ Signup"),
                key="login_signup_dash",
                horizontal=True
            )
            if login_or_signup == "🔑 Login":
                login(json_file_path)
            else:
                signup(json_file_path)

    elif page == "🔍 Mental Health Assessment":
        if st.session_state.get("logged_in"):
            st.markdown("<h2 class='gradient-header'>🔍 Mental Health Assessment</h2>", unsafe_allow_html=True)
            
            # Assessment introduction
            st.markdown("""
            <div class="card">
                <h3>How It Works</h3>
                <p>This assessment analyzes your voice patterns and/or text input to provide insights about your mental wellbeing.</p>
                <ol>
                    <li>Choose your preferred input method: voice or text</li>
                    <li>For voice input: Record or upload a voice sample where you describe how you've been feeling lately</li>
                    <li>For text input: Write about your feelings, thoughts, and recent experiences</li>
                    <li>Our AI analyzes your input and provides personalized recommendations</li>
                </ol>
                <p><em>Note: This is not a clinical diagnosis. For medical advice, please consult a healthcare professional.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reset assessment button
            if st.session_state.analysis_complete:
                if st.button("🔄 Start New Assessment", key="reset_assessment"):
                    st.session_state.voice_result = None
                    st.session_state.voice_prob = 0.0
                    st.session_state.text_result = None
                    st.session_state.audio_data = None
                    st.session_state.analysis_complete = False
                    st.session_state.show_results = False
                    st.session_state.current_diagnosis = None
                    st.session_state.current_status = None
                    st.session_state.input_method = None
                    st.session_state.text_input_data = None
                    st.rerun()
            
            # Check if models are loaded
            if not text_model_loaded:
                st.warning("⚠️ Text analysis model could not be loaded. Assessment functionality may be limited.")
            
            
            if not st.session_state.show_results:
                # Select input method
                input_options = ["🎤 Voice Input", "✏️ Text Input"]
                input_method = st.radio("Choose your preferred input method:", input_options, horizontal=True)
                st.session_state.input_method = "voice" if input_method == "🎤 Voice Input" else "text"
                
                # Voice input section
                if st.session_state.input_method == "voice":
                    st.markdown("<h3 style='margin-top: 30px;'>🎙️ Voice Input</h3>", unsafe_allow_html=True)
                    
                    if not voice_model_loaded:
                        st.warning("⚠️ Voice analysis model could not be loaded. Please use text input instead.")
                    
                    options = ["🎤 Record", "📁 Upload"]
                    choice = st.radio("Choose an option for voice input:", options, horizontal=True)
                    
                    st.markdown("""
                    <div style="background-color: #f1f5f9; padding: 15px; border-radius: 5px; margin: 15px 0;">
                        <p><strong>Suggestion:</strong> For the best results, speak for at least 30 seconds about how you've been feeling 
                        lately, your mood, energy levels, sleep patterns, and general mental state.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if choice == "🎤 Record":
                        st.write("Click the button below to start recording:")
                        audio = st_audiorec()
                        if audio is not None:
                            st.session_state.audio_data = audio
                            st.success("✅ Recording saved! Click 'Analyze Recording' to continue.")
                            
                            if st.button('🔍 Analyze Recording', key="analyze_recording"):
                                # Show a processing animation
                                with st.spinner("🔄 Processing your assessment..."):
                                    # Save audio for analysis
                                    with open("temp_analysis.wav", "wb") as f:
                                        f.write(audio)
                                    
                                    # Create a progress bar
                                    progress_bar = st.progress(0)
                                    
                                    # Analyze voice for depression
                                    progress_bar.progress(25)
                                    time.sleep(0.5)  # Simulate processing time
                                    st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                                    # Analyze emotion from the audio
                                    st.session_state.emotion_result = analyze_emotion("temp_analysis.wav", input_type="audio")
                                    progress_bar.progress(50)
                                    time.sleep(0.5)  # Simulate processing time
                                    
                                    # Initialize text result as normal (fallback)
                                    st.session_state.text_result = "normal"
                                    
                                    # Only attempt text analysis if the model was loaded properly
                                    if text_model_loaded:
                                        try:
                                            # Get transcribed text
                                            transcribed_text = transcribe_audio_from_data(audio)
                                            
                                            progress_bar.progress(75)
                                            time.sleep(0.5)  # Simulate processing time
                                            
                                            if transcribed_text:
                                                # Process text using the TF-IDF vectorizer
                                                processed_text = preprocess(transcribed_text)
                                                
                                                # Use the vectorizer to transform the text
                                                tfidf_vector = vectorizer.transform([processed_text])
                                                
                                                # Get prediction from text model
                                                text_pred = text_model.predict(tfidf_vector)
                                                st.session_state.text_result = text_pred[0]
                                        except Exception as e:
                                            # If there's any error in text analysis, silently fallback
                                            pass
                                    
                                    # Complete the progress bar
                                    progress_bar.progress(100)
                                    
                                    # Get the combined diagnosis
                                    st.session_state.current_diagnosis, st.session_state.current_status = get_combined_diagnosis(
                                        st.session_state.voice_result, 
                                        st.session_state.text_result
                                    )
                                    
                                    st.session_state.analysis_complete = True
                                    st.session_state.show_results = True
                                    
                                    # Remove temporary file
                                    if os.path.exists("temp_analysis.wav"):
                                        os.remove("temp_analysis.wav")
                                
                                # Refresh the page to show results
                                st.rerun()
                    
                    elif choice == "📁 Upload":
                        audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
                        if audio is not None:
                            st.audio(audio, format="audio/wav")
                            audio_bytes = audio.read()
                            st.session_state.audio_data = audio_bytes
                            st.success("✅ Audio uploaded! Click 'Analyze Upload' to continue.")
                            
                            if st.button('🔍 Analyze Upload', key="analyze_upload"):
                                # Show a processing animation
                                with st.spinner("🔄 Processing your assessment..."):
                                    # Save audio for analysis
                                    with open("temp_analysis.wav", "wb") as f:
                                        f.write(audio_bytes)
                                    
                                    # Create a progress bar
                                    progress_bar = st.progress(0)
                                    
                                    # Analyze voice for depression
                                    progress_bar.progress(25)
                                    time.sleep(0.5)  # Simulate processing time
                                    st.session_state.voice_result, st.session_state.voice_prob = analyze_voice("temp_analysis.wav")
                                    
                                    progress_bar.progress(50)
                                    time.sleep(0.5)  # Simulate processing time
                                    
                                    # Initialize text result as normal (fallback)
                                    st.session_state.text_result = "normal"
                                    
                                    # Only attempt text analysis if the model was loaded properly
                                    if text_model_loaded:
                                        try:
                                            # Get transcribed text
                                            transcribed_text = transcribe_audio_from_data(audio_bytes)
                                            
                                            progress_bar.progress(75)
                                            time.sleep(0.5)  # Simulate processing time
                                            
                                            if transcribed_text:
                                                # Process text using the TF-IDF vectorizer
                                                processed_text = preprocess(transcribed_text)
                                                
                                                # Use the vectorizer to transform the text
                                                tfidf_vector = vectorizer.transform([processed_text])
                                                
                                                # Get prediction from text model
                                                text_pred = text_model.predict(tfidf_vector)
                                                st.session_state.text_result = text_pred[0]
                                        except Exception as e:
                                            # If there's any error in text analysis, silently fallback
                                            pass
                                    
                                    # Complete the progress bar
                                    progress_bar.progress(100)
                                    
                                    # Get the combined diagnosis
                                    # Get the combined diagnosis with emotion
                                    st.session_state.current_diagnosis, st.session_state.current_status = get_combined_diagnosis(
                                        st.session_state.voice_result, 
                                        st.session_state.text_result,
                                        st.session_state.emotion_result
                                    )
                                                                        
                                    st.session_state.analysis_complete = True
                                    st.session_state.show_results = True
                                    
                                    # Remove temporary file
                                    if os.path.exists("temp_analysis.wav"):
                                        os.remove("temp_analysis.wav")
                                
                                # Refresh the page to show results
                                st.rerun()
                                
             
                elif st.session_state.input_method == "text":
                    st.markdown("<h3 style='margin-top: 30px;'>✏️ Text Input</h3>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style="background-color: #f1f5f9; padding: 15px; border-radius: 5px; margin: 15px 0;">
                        <p><strong>Suggestion:</strong> For the best results, write at least a few paragraphs about how you've been feeling 
                        lately, your mood, energy levels, sleep patterns, and general mental state. The more detail you provide, 
                        the more accurate our assessment will be.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    text_input = st.text_area(
                        "Please describe how you've been feeling lately:",
                        height=200,
                        max_chars=5000,
                        placeholder="I've been feeling..."
                    )
                    
                    st.session_state.text_input_data = text_input
                    
                    if st.button('🔍 Analyze Text', key="analyze_text"):
                       
                            # Debug print
                        print(f"Analyzing text: {text_input[:50]}...")  # Show first 50 chars
                        
                        # Call emotion analysis and capture the result
                        emotion_result = analyze_text_emotion(text_input, input_type="text")
                        print(f"Emotion analysis result: {emotion_result}")
                        
                        # Store in session state
                        st.session_state.emotion_result = emotion_result
                        if text_input.strip():
                        # Analyze emotion from text
                        
                        # Show a processing animation
                            with st.spinner("🔄 Processing your assessment..."):
                                # Create a progress bar
                                progress_bar = st.progress(0)
                                
                                progress_bar.progress(30)
                                time.sleep(0.5)  # Simulate processing time
                                
                                # Process text input with the model
                                if text_model_loaded:
                                    try:
                                        # Process text using the TF-IDF vectorizer
                                        processed_text = preprocess(text_input)
                                        
                                        progress_bar.progress(60)
                                        time.sleep(0.5)  # Simulate processing time
                                        
                                        # Use the vectorizer to transform the text
                                        tfidf_vector = vectorizer.transform([processed_text])
                                        
                                        # Get prediction from text model
                                        text_pred = text_model.predict(tfidf_vector)
                                        st.session_state.text_result = text_pred[0]
                                    except Exception as e:
                                        # If there's an error, set a default result
                                        st.session_state.text_result = "normal"
                                else:
                                    st.session_state.text_result = "normal"
                                
                                # Complete the progress bar
                                progress_bar.progress(100)
                                
                                # For text-only input, get diagnosis with special flag
                                st.session_state.current_diagnosis, st.session_state.current_status = get_combined_diagnosis(
                                    None,  # No voice result for text-only
                                    st.session_state.text_result,
                                    st.session_state.emotion_result,  # Pass emotion result
                                    "text_only"  # Pass text_only as input_method
                                )
                              
                                # st.session_state.current_diagnosis, st.session_state.current_status = get_combined_diagnosis(
                                #     None,  # No voice result for text-only
                                #     st.session_state.text_result,
                                #     "text_only"
                                # )
                                
                                st.session_state.analysis_complete = True
                                st.session_state.show_results = True
                            
                            
                            st.rerun()
                    else:
                         st.error("⚠️ Please enter some text before analyzing.")
            
            # Display final combined results if analysis is complete and showing results
            if st.session_state.analysis_complete and st.session_state.show_results:
                display_results_and_recommendations(
                    st.session_state.current_diagnosis,
                    st.session_state.current_status
                )
        else:
            st.warning("🔒 Please login or signup to use the mental health assessment.")
            login_or_signup = st.radio(
                "Choose an option:",
                ("🔑 Login", "✨ Signup"),
                key="login_signup_assess",
                horizontal=True
            )
            if login_or_signup == "🔑 Login":
                login(json_file_path)
            else:
                signup(json_file_path)
            
if __name__ == "__main__":
    initialize_database()
    main()