import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import requests
import google.generativeai as genai
import os
from keys import gemini_key, groq_key

# Variables for API keys
GROQ_API_KEY = groq_key
GEMINI_API_KEY = gemini_key

# Configure Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Set up the Google Gemini model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def record_audio(duration=10, sample_rate=16000):
    st.write("Recording audio... Speak now.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    audio_file = "user_input.wav"
    wavfile.write(audio_file, sample_rate, recording)
    st.write(f"Audio saved as {audio_file}")
    return audio_file

def transcribe_audio(audio_file_path):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {"model": "whisper-large-v3"}
    
    with open(audio_file_path, 'rb') as audio_file:
        files = {'file': ('user_input.wav', audio_file, 'audio/wav')}
        response = requests.post(url, headers=headers, files=files, data=data)
    
    st.write(f"Raw response: {response.text}")

    if response.status_code == 200:
        transcription = response.json().get("text", "No transcription found")
        st.write(f"Transcription: {transcription}")
        return transcription
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def generate_response(prompt):
    try:
        convo = model.start_chat(history=[])
        convo.send_message(prompt)
        response = convo.last.text
        st.write(f"Full response: {response}")
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def main():
    st.title("Voice Assistant with Google Gemini")

    st.sidebar.title("Controls")
    action = st.sidebar.radio("Choose an action", ["Record", "Quit"])

    if action == "Record":
        if st.button("Start Recording"):
            audio_file = record_audio()
            
            if audio_file:
                transcription = transcribe_audio(audio_file)
                if transcription:
                    response = generate_response(transcription)
                    if response:
                        st.write("Voice Assistant Response:")
                        st.write(response)
                else:
                    st.write("No transcription received.")
        else:
            st.write("Press the button to start recording.")
    
    elif action == "Quit":
        st.write("Exiting voice assistant...")
        st.stop()

if __name__ == "__main__":
    main()
