import streamlit as st
import os
import uuid
import base64
import json
import speech_recognition as sr
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from dotenv import load_dotenv
from groq import Groq
import librosa
import soundfile as sf
from langdetect import detect
from PIL import Image

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client for Whisper
client = Groq()

# Preprocess audio function
def preprocess_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    temp_audio_path = f"temp_{uuid.uuid4()}.wav"
    sf.write(temp_audio_path, audio, sr)
    return temp_audio_path

# Function to generate the answer for Vodafone workflow
def get_answer(chat_history, user_query, user_transaction=None, user_discount=None, user_offers=None, user_services=None):
    """
    Generate the answer based on the Vodafone workflow using the provided inputs and chat history.
    """
    unique_id = uuid.uuid4()
    
    # Prompt Template for the Workflow
    prompt_template = f"""
    Hey! I'm Vini, your Vodafone assistant. I'm here to guide you with your queries. Whether you're looking for transaction details, discounts, offers, or Vodafone services, I’ve got you covered! 😊

    ### Vodafone Assistance Instructions:

    1. **Transaction Details**:
        - I can fetch details like recent recharges, data usage, or balance.
        - For example: "What’s my last recharge?" or "How much balance do I have?"

    2. **Discounts**:
        - I’ll provide ongoing discounts on recharges, data packs, or other services.
        - For example: "Are there any recharge discounts?" or "Show me discounted data packs."

    3. **Offers**:
        - I can help with seasonal or personalized offers.
        - For example: "What offers do I have?" or "Are there any special offers for me?"

    4. **Services**:
        - Need help activating/deactivating services like IR packs, caller tunes, or Vi apps? I can guide you.
        - For example: "Activate international roaming" or "Show me Vi app features."

    ### Chat History:
    {chat_history}

    ### User Query:
    {user_query}

    ### Additional Information:
    - User Transaction: {user_transaction}
    - User Discounts: {user_discount}
    - User Offers: {user_offers}
    - User Services: {user_services}

    """
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": prompt_template}],
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            stream=False,
        )
        
        # Retrieve the response
        response_text = completion.choices[0].message.content.strip()
        return response_text
    
    except Exception as e:
        print("Error occurred:", e)
        return "An error occurred while processing your request."# Speech-to-text function using Whisper model
def speech_to_text(audio_path):
    processed_audio_path = preprocess_audio(audio_path)
    with open(processed_audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    recognized_text = transcription.text if hasattr(transcription, 'text') else ""
    try:
        detected_language = detect(recognized_text)
        print(f"Detected language: {detected_language}")
    except Exception as e:
        detected_language = 'unknown'
    return recognized_text if detected_language in ['en', 'unknown'] else ""

# Convert text to speech
def text_to_speech(input_text):
    tts = gTTS(text=input_text, lang='en')
    wav_file_path = "temp_audio_play.mp3"
    tts.save(wav_file_path)
    return wav_file_path

# Autoplay audio in Streamlit
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "✨ Hey there! I’m Vini, your Vodafone assistant, here to help with transactions, discounts, offers, and services—just ask, I’ve got you covered! ✨" }
        ]
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "user_transaction" not in st.session_state:
        st.session_state.user_transaction = None
    if "user_discount" not in st.session_state:
        st.session_state.user_discount = None
    if "user_offers" not in st.session_state:
        st.session_state.user_offers = None
    if "user_services" not in st.session_state:
        st.session_state.user_services = None
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

initialize_session_state()

# Login Functionality
def login():
    if st.session_state.name and st.session_state.number:
        st.session_state.logged_in = True
    else:
        st.error("Please enter both your name and number!")

# Logout Functionality
def logout():
    st.session_state.logged_in = False

# Custom CSS for Login Screen Styling
def custom_login_css():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(to right, #4facfe, #00f2fe);
        }
        h1 {
            color: #4CAF50;
            text-align: center;
            font-size: 3rem;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }
        .stTextInput input {
            border: 3px solid #0a0a0a;
            border-radius: 12px;
            padding: 10px;
            box-shadow: inset 0px 3px 6px rgba(0, 0, 0, 0.2), 
                        0px 2px 4px rgba(0, 0, 0, 0.1);
        }
        .stButton button {
            background-color: #32cd32;
            color: white;
            font-size: 1.2rem;
            border: none;
            border-radius: 12px;
            padding: 10px 20px;
            cursor: pointer;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stButton button:hover {
            transform: scale(1.05);
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3);
        }
        .login-inputs {
            display: flex;
            justify-content: center;
            gap: 20px;
        }
        .login-inputs .stTextInput {
            width: 45%;
        }
        .password-container {
            position: relative;
        }
        .password-container .stTextInput input {
            padding-right: 40px;
        }
        .password-container .eye-icon {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Display warning message instead of pop-up with close button
def display_welcome_warning():
    if 'name' in st.session_state and 'warning_closed' not in st.session_state:
        # Display the warning
        st.warning(f"Welcome, {st.session_state.name}! Your current plan will expire on 11-26-2024.🎉", icon="⚠️")
        
        # Add a close button for the warning
        if st.button("x"):
            st.session_state.warning_closed = True

# Voice Interface
def voice_interface():
    st.markdown("<h1 style='text-align: center;'>ᴠɪɴɪ📲</h1>", unsafe_allow_html=True)
    footer_container = st.container()
    with footer_container:
        if st.session_state.audio_bytes is None:
            st.session_state.audio_bytes = audio_recorder(text=None, icon_size="3x", sample_rate=16000)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.audio_bytes:
        with st.spinner("Processing your request..."):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(st.session_state.audio_bytes)
            transcript = speech_to_text(webm_file_path)
            if transcript:
                st.session_state.messages.append({"role": "user", "content": transcript})
                st.session_state.user_query = transcript
                response = get_answer(
                    chat_history=st.session_state.messages,
                    user_query=st.session_state.user_query,
                    user_transaction=st.session_state.user_transaction,
                    user_discount=st.session_state.user_discount,
                    user_offers=st.session_state.user_offers,
                    user_services=st.session_state.user_services,
                )
                audio_path = text_to_speech(response)
                autoplay_audio(audio_path)
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                os.remove(webm_file_path)

    footer_container.float("bottom: 0rem;right: 10px;")

# Main App Logic
if not st.session_state.logged_in:
    custom_login_css()
    st.markdown("<div style='text-align: center; margin: auto; padding: 2rem;'>", unsafe_allow_html=True)
    st.markdown("<h1> 𝚆𝚎𝚕𝚌𝚘𝚖𝚎 𝚝𝚘 𝚅𝚒𝚗𝚒!</h1>", unsafe_allow_html=True)
    st.session_state.name = st.text_input("Enter your Name", placeholder="Enter your name")
    st.session_state.number = st.text_input("Enter your Number", placeholder="Enter your mobile number")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Call login directly via on_click
    st.button("Login", on_click=login)
else:
    display_welcome_warning()  # Display the warning with the close button
    voice_interface()

#
