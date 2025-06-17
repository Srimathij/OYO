import streamlit as st
import os
import uuid
import base64
import json
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from dotenv import load_dotenv
from groq import Groq
import librosa
import soundfile as sf
from langdetect import detect
import torch
from elevenlabs import ElevenLabs

# Load API key from environment variables (if using .env)
# load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "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")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize Eleven Labs client
eleven_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY", "sk_91edb9f0c2ad42ce41c3d7feb08c9a3e4615ab6ff862acf8"),
)

# Function to generate speech in various languages using Eleven Labs
# You need to set appropriate voice IDs for languages
VOICE_IDS = {
    'en': '21m00Tcm4TlvDq8ikWAM',  # English default
    'id': 'cgSgspJ2msm6clMCkdW9',   # Malay/Indonesian
    'ar': 'ZF6FPAbjXT4488VcRRnw',  # Replace with actual Arabic voice ID from ElevenLabs
    # Add other language voice IDs if needed
}


def text_to_speech(input_text, language='en'):
    voice_id = VOICE_IDS.get(language, VOICE_IDS['en'])
    audio_generator = eleven_client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        text=input_text,
    )
    audio_file_path = f"{uuid.uuid4()}.mp3"
    with open(audio_file_path, "wb") as file:
        for chunk in audio_generator:
            file.write(chunk)
    return audio_file_path

# Function to autoplay audio in Streamlit
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

# Preprocess audio function
def preprocess_audio(audio_path):
    # Load the audio file using librosa
    audio, sr = librosa.load(audio_path, sr=16000)

    # Save preprocessed audio to a temporary file
    temp_audio_path = f"temp_{uuid.uuid4()}.wav"
    sf.write(temp_audio_path, audio, sr)  # Save using soundfile

    return temp_audio_path

@st.cache_data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load rooms.json
json_file_path = "rooms.json"
rooms_data = load_json(json_file_path)

def search_rooms(location, budget_range):
    min_budget, max_budget = budget_range
    results = [room for room in rooms_data if room['Location'].lower() == location.lower() and min_budget <= room['Budget'] <= max_budget]
    return results

# Modify get_answer to accept detected_language

def get_answer(chat_history, user_query, user_location, user_budget, detected_language='en'):
    unique_id = uuid.uuid4()

    room_options = ""
    if user_location and user_budget:
        matching_rooms = search_rooms(user_location, user_budget)
        if matching_rooms:
            rooms_list = "\n".join([
                f" {room['Hotel_name']} in {room['Location']} (₹{room['Budget']}/night, {room['Amenities']}, Rating: {room['Rating']}⭐, Discount: {room['Discount']})"
                for room in matching_rooms])
            room_options = f"Here are some hotel options for {user_location} within your budget:\n\n{rooms_list}"
        else:
            room_options = f"Sorry, we couldn't find any hotel options in {user_location} within your budget."

    # Set language instruction based on detected_language
    if detected_language == 'id':
        language_instruction = "Respond strictly in **Bahasa Malaysia**."
    elif detected_language == 'ar':
        language_instruction = "Respond strictly in **Arabic**."
    else:
        language_instruction = "Respond in English."

    prompt_template = f"""
    ### CONTEXT: Insurance Assistant Bot for KGISL Insu

    You are an intelligent, courteous, and policy-aware virtual assistant for KGISL Insu. Your name is Ava. You assist users in a professional and friendly tone, providing accurate information and reminders related to their insurance policy. Your responsibilities include confirming identity, checking policy due dates, offering reminders, and guiding the user through available next steps such as payment options or customer support.

    ### PRIMARY TASKS:
    1. Greet the user and verify if you’re speaking with the policyholder.
    2. If the identity is confirmed, check the policy due date.
    3. If the due date is in the future, remind them of the upcoming payment.
    4. If the due date has passed, inform them politely and provide the grace period.
    5. Offer to assist with payment methods or redirect to support channels as needed.
    6. If the user already paid, acknowledge politely and explain any automatic processing.
    7. If the user is unavailable, incorrect, or declines to talk, end professionally.

    ### INPUTS:
    - Chat History:
    {chat_history}

    - User Query:
    {user_query}

    - Language Instruction (if applicable):
    {language_instruction}

    - Room Options (e.g., {room_options}) for routing, policy tags, or intent classification.

    ### BOT INSTRUCTIONS:
    - Always confirm user identity before sharing policy information.
    - Use clear and empathetic language, adapting to whether the user is cooperative, confused, or resistant.
    - Structure responses with natural transitions (no abrupt shifts).
    - If discussing payment:
    - Include policy name, last 4 digits of policy number, due date, premium amount, and grace period if overdue.
    - Offer payment options or point to the official KGISL portal or mobile app.
    - Maintain policy confidentiality and never expose sensitive details unless the identity is confirmed.
    - If user confirms payment:
    - Say “Thank you for your update. Please disregard this reminder. If enrolled in auto-pay, ensure your funding source is active.”

    ### SAMPLE OUTPUT FORMAT:
    Respond in a friendly, professional tone:
    - Greet ➤ Confirm identity ➤ Check due date ➤ Inform/remind ➤ Offer help or payment links ➤ Close call/chat politely.
    - Respect user’s language preference and switch languages if directed in {language_instruction}.
    - If user declines to continue, say: “Thank you for your time. Feel free to reach out to us anytime for assistance.”

    ### STYLE:
    Use short paragraphs, human-like language, and avoid overly robotic phrasing. Maintain professionalism and empathy throughout.
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

        response_text = completion.choices[0].message.content.strip()
        return response_text

    except Exception as e:
        print("Error occurred:", e)
        return "An error occurred while processing your request."

# Updated speech-to-text function to handle English, Malay, Arabic, etc.
def speech_to_text(audio_path):
    processed_audio_path = preprocess_audio(audio_path)

    with open(processed_audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json",
        )

    recognized_text = transcription.text if hasattr(transcription, 'text') else ""

    if not recognized_text:
        print("No text recognized.")
        return "", "unknown"

    try:
        detected_language = detect(recognized_text)
        print(f"Detected language: {detected_language}")
    except Exception as e:
        print(f"Language detection failed: {e}")
        detected_language = 'unknown'

    # Supported languages: Malay (id, ms), Arabic (ar), Urdu (ur), Tamil (ta), Bengali (bn), English (en)
    supported = ['id', 'ms', 'ar', 'ur', 'ta', 'bn', 'en']
    if detected_language not in supported:
        print("Unsupported language detected.")
        return "", "unsupported"

    print(f"User said: {recognized_text}")
    return recognized_text, detected_language

float_init()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "أهلاً! أنت تتحدث مع Ava. مساعدك الافتراضي في KGISL Insu. من أجل الشفافية، تعتمد هذه المكالمة على الذكاء الاصطناعي وقد يتم تسجيلها لأغراض الجودة والتدريب. هل يمكن أن تخبرني باسمك؟"}
        ]
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "user_location" not in st.session_state:
        st.session_state.user_location = None
    if "user_budget" not in st.session_state:
        st.session_state.user_budget = (0, float("inf"))
    if "detected_language" not in st.session_state:
        st.session_state.detected_language = 'en'

initialize_session_state()

# Center the header with refined styles using Streamlit's HTML rendering
st.markdown("""
    <style>
    .header {
        text-align: center;
        font-size: 2rem;  /* Medium size header */
        font-weight: normal;  /* No bold */
        margin-top: 15px;
        color: linear-gradient(135deg, #f3ec78, #af4261); /* Gradient for contrast */
    }
    .mic-container {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
    }
    body {
        background-color: #000; /* Black background */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>VoiceBot.AI✨</div>", unsafe_allow_html=True)

# Create footer container for the microphone
footer_container = st.container()

# Initialize the user's query to None
user_query = None

# Handle audio input in the footer container
with footer_container:
    transcript = None

    audio_bytes = audio_recorder(
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone-lines",
        icon_size="3x",
    )

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    with st.spinner("Hang on! I’m finding the best options for you...⏳"):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        transcript, detected_language = speech_to_text(webm_file_path)
        st.session_state.detected_language = detected_language
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            st.session_state.user_query = transcript

            # Detect if user mentioned a location
            if "in " in transcript.lower():
                st.session_state.user_location = transcript.split("in ")[-1].strip()

            # Detect if user mentioned a budget range
            if "between ₹" in transcript.lower():
                try:
                    budget_part = transcript.split("between ₹")[-1].split(" and ")
                    min_budget = int(budget_part[0].strip().replace(",", ""))
                    max_budget = int(budget_part[1].strip().replace(",", ""))
                    st.session_state.user_budget = (min_budget, max_budget)
                except:
                    pass

            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            if st.session_state.user_query:
                final_response = get_answer(
                    st.session_state.messages,
                    st.session_state.user_query,
                    st.session_state.user_location,
                    st.session_state.user_budget,
                    st.session_state.detected_language
                )
            else:
                final_response = get_answer(
                    st.session_state.messages,
                    st.session_state.user_query,
                    st.session_state.user_location,
                    st.session_state.user_budget,
                    st.session_state.detected_language
                )
        with st.spinner("Creating the perfect answer for you...✨"):
            audio_file = text_to_speech(final_response, language=st.session_state.detected_language)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

# Float the footer container
footer_container.float("bottom: 0rem; right: 10px;")
