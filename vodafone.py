import streamlit as st
import os
import uuid
import base64
import json
import speech_recognition as sr
import pyttsx3 
from audio_recorder_streamlit import audio_recorder
import streamlit.components.v1 as components
from streamlit_float import *
from dotenv import load_dotenv
from groq import Groq
from streamlit_carousel import carousel
from streamlit_card import card
import librosa
import soundfile as sf
from langdetect import detect
from PIL import Image

# Initialize session state
def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "✨ Hey there! I’m Vini, your Vodafone assistant, here to help with transactions, discounts, offers, and services—just ask, I’ve got you covered! ✨\n\n🎉 Today’s deal: Recharge with ₹300 or more and get 10% cashback! Don't miss out on this amazing offer. 🎉\n\n🔥 Limited time offer: Get 20% off on your next mobile plan when you upgrade today! 🔥" }
        ]
   
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
   
    if "feedbacks" not in st.session_state:
        st.session_state.feedbacks = []  # Store feedback data
    if "user_transaction" not in st.session_state:
        st.session_state.user_transaction = None
    if "user_discount" not in st.session_state:
        st.session_state.user_discount = None
    if "user_offers" not in st.session_state:
        st.session_state.user_offers = None
    if "user_services" not in st.session_state:
        st.session_state.user_services = None
    if "user_location" not in st.session_state:
        st.session_state.user_location = None
    if "user_budget" not in st.session_state:
        st.session_state.user_budget = (0, float("inf"))
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

initialize_session_state()



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


def get_answer(chat_history, user_query, user_transaction=None, user_discount=None, user_offers=None, user_services=None):
    """
    Generate the answer based on the Vodafone workflow using the provided inputs and chat history.
    Detects language to respond in Hindi if required.
    """
    unique_id = uuid.uuid4()

    # Detect language
    language = detect_language(user_query)  # Function to detect the language of the query (defined below)

    if language == "hi":  
        prompt_template = f"""
        नमस्ते! मैं विनी हूँ, आपकी वोडाफोन सहायक। 😊 मैं आपकी मदद करने के लिए यहाँ हूँ। चलिए आपके सवालों का समाधान ढूंढते हैं।

        ### वोडाफोन कार्यप्रवाह:

        1. **लॉगिन सहायता**:
            - उपयोगकर्ता अपने फोन नंबर या उपयोगकर्ता नाम और पासवर्ड के साथ लॉगिन करते हैं।
            - अगर लॉगिन सफल होता है, तो मैं व्यक्तिगत रूप से उपयोगकर्ता का स्वागत करती हूँ, हाल की गतिविधियों का उल्लेख करती हूँ और ऑफर्स पर प्रकाश डालती हूँ।
            - अगर लॉगिन असफल होता है, तो मैं पुनर्प्राप्ति विकल्प या सहायता संपर्क विवरण प्रदान करती हूँ।

        2. **व्यक्तिगत अभिवादन**:
            - लॉगिन के बाद, मैं उपयोगकर्ता का नाम लेकर स्वागत करती हूँ।
            - मैं उनकी प्राथमिकताओं या पिछले इंटरैक्शन के आधार पर विशेष ऑफर्स का उल्लेख करती हूँ, जैसे:
            - "₹300 या उससे अधिक का रिचार्ज करें और 10% कैशबैक पाएं!"

        3. **प्लान स्थिति**:
            - मैं वर्तमान रिचार्ज प्लान की स्थिति की जांच करती हूँ:
            - अगर प्लान समाप्त हो गया है, तो मैं समाप्त प्लान प्रक्रिया में मार्गदर्शन करती हूँ।
            - अगर प्लान सक्रिय है, तो मैं समाप्ति की जानकारी प्रदान करती हूँ और अपग्रेड विकल्प दिखाती हूँ।

        4. **रिचार्ज प्लान्स**:
            - मैं कस्टमाइज्ड रिचार्ज प्लान्स की सूची प्रस्तुत करती हूँ, जैसे:
            - प्लान A: ₹199 | 1GB/दिन | 28 दिन।
            - प्लान B: ₹299 | अनलिमिटेड कॉल्स + 2GB/दिन | 56 दिन।
            - प्लान्स में लागत, अवधि, और लाभ के विवरण होते हैं।

        5. **प्लान चयन और भुगतान**:
            - मैं उपयोगकर्ताओं को प्लान चुनने में मदद करती हूँ और सुरक्षित भुगतान विकल्पों जैसे UPI या क्रेडिट कार्ड के माध्यम से मार्गदर्शन करती हूँ।

        6. **पुष्टि और अपडेट्स**:
            - भुगतान के बाद, मैं सफल संदेश भेजती हूँ जैसे:
            - "सफलता! आपका नया प्लान अब सक्रिय है।"
            - मैं उपयोगकर्ताओं को प्लान की समाप्ति की याद दिलाती हूँ और उपयोग के आधार पर अपग्रेड सुझाव देती हूँ।

        7. **डिस्काउंट्स और ऑफर्स**:
            - अगर उपयोगकर्ता छूट या डील्स के बारे में पूछते हैं, तो मैं उनके लिए विशेष डील्स की सूची दिखाती हूँ।
            - प्रत्येक कार्ड में शामिल है:
                - **शीर्षक**: ऑफर का नाम, जैसे "10% कैशबैक पर रिचार्ज।"
                - **विवरण**: मुख्य विवरण, जैसे "₹300 या अधिक के रिचार्ज पर कैशबैक।"
                - **चित्र**: ऑफर का विज़ुअल।
                - **वैधता**: ऑफर की समाप्ति तिथि।
            - उदाहरण:
                - **शीर्षक**: "डबल डेटा ऑफर!"
                - **विवरण**: "₹499 के रिचार्ज पर 30 दिनों के लिए डबल डेटा।"
                - **वैधता**: "31 दिसंबर 2024 तक वैध।"

        ### चैट इतिहास:
        {chat_history}

        ### उपयोगकर्ता प्रश्न:
        {user_query}

        ### अतिरिक्त संदर्भ:
        - लेन-देन: {user_transaction}
        - ऑफर्स: {user_offers}
        - सेवाएं: {user_services}
        """
    else: 
        prompt_template = f"""
        Hey there! I'm Vini, your Vodafone assistant, here to assist you exclusively with Vodafone-related queries. Let’s dive into your questions and explore how I can assist you today.

        ### Vodafone Workflow:

        1. **Login Assistance**:
            - Users log in with their phone number or username and password.
            - If login succeeds, I greet the user personally, acknowledge recent activity, and highlight Vodafone offers.
            - If login fails, I provide account recovery options or support contact details.

        2. **Personalized Greetings**:
            - After login, I warmly welcome the user by name.
            - I highlight Vodafone-specific offers based on preferences or past interactions, such as:
                - "Recharge with ₹300 or more and get 10% cashback!"

        3. **Plan Status**:
            - I check the status of the current Vodafone recharge plan:
                - If expired, I guide the user through the Expired Plan Flow.
                - If active, I provide details about expiry and highlight potential Vodafone upgrades.

        4. **Recharge Plans**:
            - I present a list of Vodafone-specific recharge plans, such as:
                - Plan A: ₹199 | 1GB/day | 28 days.
                - Plan B: ₹299 | Unlimited calls + 2GB/day | 56 days.
            - Plans include cost, duration, and benefits for easy comparison.

        5. **Plan Selection & Payment**:
            - I help users choose a Vodafone plan, provide details if needed, and guide them through secure payment options like UPI or credit cards.

        6. **Confirmation & Updates**:
            - After payment, I confirm success with messages like:
                - "Success! Your new Vodafone plan is now active."
            - I remind users of their plan's expiry and suggest upgrades based on usage.

        7. **Reminders & Notifications**:
            - If requested, I set reminders for Vodafone plan expiry:
                - "Would you like a reminder a day before your plan expires?"
            - I schedule the reminder and confirm it.

        8. **Discounts & Offers**:
            - If the user asks about Vodafone discounts, offers, or deals, I provide a carousel of exciting Vodafone deals tailored for them.
            - Each carousel card includes:
                - **Title**: Offer name, such as "10% Cashback on Recharge."
                - **Description**: Key details, such as "Recharge with ₹300 or more to avail cashback."
                - **Image**: A visual representation of the deal.
                - **Validity**: The expiration date of the offer.
            - For example:
                - **Title**: "Double Data Offer!"
                - **Description**: "Recharge ₹499 and get double data for 30 days."
                - **Validity**: "Valid until 31st Dec 2024."
            - I ensure the carousel provides a user-friendly view for comparison and selection.

        ### Out-of-Scope Handling:

        - If a user asks a question unrelated to Vodafone products or services, I respond with:
            - "I’m here to assist with Vodafone-related queries only. For unrelated topics, I recommend consulting the relevant resources or support."


        ### Chat History:
        {chat_history}

        ### User Query:
        {user_query}

        ### Additional Context:
        - Transactions: {user_transaction}
        - Offers: {user_offers}
        - Services: {user_services}
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
        return "An error occurred while processing your request."


def detect_language(text):
    """
    A function to detect the language of the given text.
    It integrates the `langdetect` library to identify the language.
    """
    from langdetect import detect
    try:
        lang_code = detect(text)  # Detects the language code (e.g., 'en', 'hi', etc.)
        
        # Additional logic for Hindi
        if lang_code == 'hi':
            print("Detected language: Hindi")
            return 'hi'
        elif lang_code == 'en':
            print("Detected language: English")
            return 'en'
        else:
            print(f"Detected language: {lang_code} (Unsupported)")
            return lang_code  # Return other detected languages if necessary
    except Exception as e:
        print("Language detection failed:", e)
        return "en"  # Default to English on failure


from langdetect import detect  # Importing langdetect library for language detection

def speech_to_text(audio_path):
    """
    Converts speech from audio to text and detects the language of the transcription.

    Args:
        audio_path (str): Path to the audio file to be transcribed.

    Returns:
        str: The recognized text if the detected language is English or Hindi, else an empty string.
    """
    try:
        # Preprocess the audio (assuming preprocess_audio function is already defined)
        processed_audio_path = preprocess_audio(audio_path)

        # Open the processed audio file
        with open(processed_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="verbose_json",
            )

        # Extract the recognized text from the transcription
        recognized_text = transcription.text if hasattr(transcription, 'text') else ""

        # Detect the language of the recognized text
        detected_language = detect(recognized_text)
        print(f"Detected language: {detected_language}")

        # Return the recognized text only if the language is English or Hindi
        if detected_language in ['en', 'hi']:
            return recognized_text
        else:
            return ""  # Return empty string for unsupported languages

    except Exception as e:
        print(f"Error during speech-to-text conversion: {e}")
        return ""
# from gtts import gTTS

# Convert text to speech with customizable tld and language
import pyttsx3
import os
from langdetect import detect, DetectorFactory

# Ensure consistent language detection
DetectorFactory.seed = 0

def text_to_speech(input_text, default_lang="en", default_tld="co.in", slow=False, output_file="temp_audio_play.mp3"):
    """
    Converts text to speech using pyttsx3 with English-only support.

    Args:
        input_text (str): The text to convert to speech.
        default_lang (str): The default language code (default is "en").
        default_tld (str): The default top-level domain for regional accent (default is "co.in").
        slow (bool): Whether the speech should be slower (default is False).
        output_file (str): The file path to save the audio (default is "temp_audio_play.mp3").

    Returns:
        str: The file path to the saved audio, or None if an error occurs.
    """
    try:
        # Detect the language of the input text
        detected_lang = detect(input_text)
        print(f"Detected language: {detected_lang}")  # Debug information

        if detected_lang != "en":
            print("Detected non-English language. Defaulting to English.")

        # Initialize pyttsx3 engine
        engine = pyttsx3.init()

        # Adjust speech rate
        base_rate = 160  # Conversational rate
        engine.setProperty('rate', base_rate - 40 if slow else base_rate)

        # Adjust volume for clarity
        engine.setProperty('volume', 0.9)

        # Set voice for English
        voices = engine.getProperty('voices')
        print(f"Available voices: {[voice.id for voice in voices]}")  # Debug information

        # Use English voice
        if default_tld == "co.in" and len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
            print(f"Using Indian English voice: {voices[1].id}")
        else:
            engine.setProperty('voice', voices[0].id)
            print("Using default English voice as fallback.")

        # Ensure the output path is valid
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory for audio file: {output_dir}")

        # Save the audio to a file
        engine.save_to_file(input_text, output_file)
        engine.runAndWait()
        print(f"Audio saved to file: {output_file}")

        return output_file
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        return None

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

import json

# Load plans.json data
def load_plans():
    with open("plans.json", "r") as file:
        return json.load(file)

plans_data = load_plans()

# Modify login functionality to include plan status
def login():
    if st.session_state.name and st.session_state.number:
        for user in plans_data["users"]:
            if (
                user["name"].lower() == st.session_state.name.lower()
                and user["number"] == st.session_state.number
            ):
                st.session_state.logged_in = True
                st.session_state.plan_status = user["plan_status"]
                return
        st.error("User not found. Please check your name and number.")
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

# Callback function to close the warning
def close_warning():
    st.session_state.warning_closed = True

# Display warning message with close button based on plan status
def display_welcome_warning():
    if 'name' in st.session_state and not st.session_state.get('warning_closed', False):
        # Create two columns: one for the message and one for the close button
        cols = st.columns([8, 1])  # Adjust column sizes as needed
        
        # Determine the message based on the plan status
        if st.session_state.plan_status == "active":
            message = f"Hello, {st.session_state.name}! Your plan is currently active and valid until 11-26-2024. ✅"
            message_type = "success"
        elif st.session_state.plan_status == "expired":
            message = f"Hello, {st.session_state.name}! Your plan has expired. Please recharge as soon as possible. ⚠️"
            message_type = "warning"
        else:
            message = f"Hello, {st.session_state.name}! Unable to determine your plan status. Please contact support. ❓"
            message_type = "info"
        
        # Display the appropriate message
        with cols[0]:
            if message_type == "success":
                st.success(message, icon="✅")
            elif message_type == "warning":
                st.warning(message, icon="⚠️")
            else:
                st.info(message, icon="ℹ️")
        
        # Add close button
        with cols[1]:
            st.button("x", key="close_warning", on_click=close_warning)

# Callback function to close the warning
def close_warning():
    st.session_state.warning_closed = True

# Display warning message based on plan status
def display_plan_warning():
    if st.session_state.plan_status == "active":
        st.success(f"Hello, {st.session_state.name}! Your plan is currently active. ✅")
    elif st.session_state.plan_status == "expired":
        st.warning(f"Hello, {st.session_state.name}! Your plan has expired. Please recharge. ⚠️")


# Import additional libraries for the carousel
# from itertools import cycle

from streamlit_carousel import carousel

# Detect if the query is related to offers/deals
def is_offer_query(user_query):
    keywords = ["plans", "plan", "plan's"]
    return any(keyword in user_query.lower() for keyword in keywords)

# Sample offers data (can be replaced with database/API data)
offers_data = [
    {
        "id": 1,
        "title": "𝗣𝗟𝗔𝗡 𝗔",
        "description": "1𝐆𝐁/𝐝𝐚𝐲 𝐟𝐨𝐫 ₹199!",
        "validity": "𝐯𝐚𝐥𝐢𝐝 𝐟𝐨𝐫 28 𝐝𝐚𝐲𝐬",
        "image": "https://cdn.worldvectorlogo.com/logos/vodafone-business-1.svg",  # Replace with actual image URL
    },
    {
        "id": 2,
        "title": "𝗣𝗟𝗔𝗡 𝗕",
        "description": "𝗨𝗻𝗹𝗶𝗺𝗶𝘁𝗲𝗱 𝗰𝗮𝗹𝗹𝘀 + 2𝗚𝗕/𝗱𝗮𝘆 𝗳𝗼𝗿 ₹299",
        "validity": "𝘃𝗮𝗹𝗶𝗱 𝗳𝗼𝗿 56 𝗱𝗮𝘆𝘀!",
        "image": "https://cdn.worldvectorlogo.com/logos/vodafone-business-1.svg",  # Replace with actual image URL
    },
   
]

# Function to display the carousel
def display_carousel(offers_data):
    # Create carousel items from offers data
    offer_carousel_items = [
        dict(
            title=offer["title"],
            text=f"{offer['description']} (Valid: {offer['validity']})",
            img=offer["image"],  # Use the image URL for each offer
        )
        for offer in offers_data
    ]
    
    # Display the carousel with the generated offer items
    carousel(items=offer_carousel_items)


########Offers#########

from streamlit_card import card

# Detect if the query is related to offers/deals
def is_off_query(user_query):
    keywords = ["offers", "offer", "offer's"]
    return any(keyword in user_query.lower() for keyword in keywords)

# Sample offers data (can be replaced with database/API data)
off_data = [
    {
        "id": 1,
        "title": "ᴜɴʟɪᴍɪᴛᴇᴅ ᴄᴀʟʟꜱ ꜰᴏʀ 56 ᴅᴀʏꜱ",
        "description": "𝘙𝘦𝘤𝘩𝘢𝘳𝘨𝘦 ₹399 𝘢𝘯𝘥 𝘦𝘯𝘫𝘰𝘺 𝘶𝘯𝘭𝘪𝘮𝘪𝘵𝘦𝘥 𝘤𝘢𝘭𝘭𝘴 + 2𝘎𝘉/𝘥𝘢𝘺.",
        "validity": "Valid until 12-15-2024",
        "image": "https://www.myvi.in/content/dam/vodafoneideadigital/homespyder/Vi-logo.svg",  # Replace with actual image URL
    }
    
]

# Function to display cards for each offer
from streamlit_card import card

def display_cards(off_data):
    for offer in off_data:
        hasClicked = card(
            title=offer["title"],
            text=f"{offer['description']} (Valid: {offer['validity']})",
            image=offer["image"],  # Use the image URL for each offer
            url="https://www.myvi.in/prepaid/best-prepaid-plans",  # You can add a URL here if you want the card to be clickable
        )
        # Optionally handle the click event for each card
        if hasClicked:
            st.write("This is the perfect offer for you!✨ If you're looking for more amazing deals like this, let us know! We're here to help you explore. 🚀")

# import streamlit as st
# from streamlit_feedback import streamlit_feedback


# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


# def display_answer():
#     for i in st.session_state.chat_history:
#         with st.chat_message("human"):
#             st.write(i["question"])
#         with st.chat_message("ai"):
#             st.write(i["answer"])

#         # If there is no feedback show N/A
#         if "feedback" in i:
#             st.write(f"Feedback: {i['feedback']}")
#         else:
#             st.write("Feedback: N/A")

# def create_answer(question):
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     message_id = len(st.session_state.chat_history)

#     st.session_state.chat_history.append({
#         "question": question,
#         "answer": f"{question}_Answer",
#         "message_id": message_id,
#     })


# def fbcb():
#     message_id = len(st.session_state.chat_history) - 1
#     if message_id >= 0:
#         st.session_state.chat_history[message_id]["feedback"] = st.session_state.fb_k
#     display_answer()


# if question := st.chat_input(placeholder="Ask your question here .... !!!!"):
#     create_answer(question)
#     display_answer()

#     with st.form('form'):
#         streamlit_feedback(feedback_type="thumbs", align="flex-start", key='fb_k')
#         st.form_submit_button('Save feedback', on_click=fbcb)

###### Voice interface -------------------------



import streamlit as st
from streamlit_feedback import streamlit_feedback


def voice_interface():
    # Initialize session state variables
    if "feedback_processed" not in st.session_state:
        st.session_state.feedback_processed = []
    if "is_speaking" not in st.session_state:
        st.session_state.is_speaking = False  # Tracks if the bot is currently speaking
    if "stop_clicked" not in st.session_state:
        st.session_state.stop_clicked = False  # Tracks if the stop button is clicked

    st.markdown("<h3 style='text-align: center;'>𝐕𝐎𝐃𝐀𝐅𝐎𝐍𝐄.𝐀𝐈</h3>", unsafe_allow_html=True)

    # Footer for the audio recorder
    footer_container = st.container()
    with footer_container:
        audio_bytes = audio_recorder(text=None, icon_size="3x", sample_rate=16000)
        stop_button = st.button("Stop speaking 🔇")  # Add Stop Speaking button

        # Handle stop speaking action
        if stop_button:
            st.session_state.is_speaking = False
            st.session_state.stop_clicked = True
            st.warning("Voice playback stopped. You can record again.")

    # Display all messages with feedback for assistant responses
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Add feedback form for assistant messages
            if message["role"] == "assistant" and index not in st.session_state.feedback_processed:
                feedback_key = f"feedback_{index}"  # Unique key for feedback widget
                with st.form(key=f"feedback_form_{index}"):
                    feedback = streamlit_feedback(feedback_type="thumbs", align="flex-start", key=feedback_key)
                    submitted = st.form_submit_button("Submit Feedback")
                    if submitted and feedback is not None:
                        # Record feedback and mark as processed
                        st.session_state.feedback_processed.append(index)
                        st.session_state.feedbacks.append({
                            "message_id": index,
                            "message": message["content"],
                            "feedback": feedback
                        })
                        st.success("Thank you for your feedback!")

    # Handle audio input
    if audio_bytes:
        with st.spinner("Hang on! I’m finding the best options for you...⏳"):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            # Convert speech to text
            transcript = speech_to_text(webm_file_path)
            if transcript:
                st.session_state.messages.append({"role": "user", "content": transcript})
                st.session_state.user_query = transcript
                st.session_state.response_generated = False  # Reset response generation flag

                with st.chat_message("user"):
                    st.write(transcript)
                os.remove(webm_file_path)

    
    # Generate assistant response only if not already generated
    if st.session_state.user_query and not st.session_state.response_generated:
        # Avoid response generation if feedback is already processed
        if st.session_state.user_query not in st.session_state.feedback_processed:
            if is_offer_query(st.session_state.user_query):
                # Handle offer-related queries
                response = "🎉 Exciting Plans Just for You! 🎉 Explore our amazing offers tailored to meet your needs. Let's find the perfect plan for you! ✨"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.response_generated = True  # Mark response as generated

                # Render assistant message and feedback immediately
                with st.chat_message("assistant"):
                    st.write(response)
                    feedback_key = f"feedback_{len(st.session_state.messages) - 1}"  # Unique key for feedback
                    feedback = streamlit_feedback(feedback_type="thumbs", align="flex-start", key=feedback_key)
                    if feedback:
                        st.session_state.feedbacks.append({
                            "message_id": len(st.session_state.messages) - 1,
                            "message": response,
                            "feedback": feedback
                        })
                display_carousel(offers_data)

            elif is_off_query(st.session_state.user_query):
                # Handle off-related queries
                response = "Amazing Offers Await You!🌟 Dive into our exclusive deals and discover the perfect one for you. Let’s get started! 🚀"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.response_generated = True  # Mark response as generated

                # Render assistant message and feedback immediately
                with st.chat_message("assistant"):
                    st.write(response)
                    feedback_key = f"feedback_{len(st.session_state.messages) - 1}"  # Unique key for feedback
                    feedback = streamlit_feedback(feedback_type="thumbs", align="flex-start", key=feedback_key)
                    if feedback:
                        st.session_state.feedbacks.append({
                            "message_id": len(st.session_state.messages) - 1,
                            "message": response,
                            "feedback": feedback
                        })
                display_cards(off_data)

            else:
                # Handle general queries
                with st.spinner("Thinking🤔..."):
                    final_response = get_answer(
                        st.session_state.messages,
                        st.session_state.user_query,
                        st.session_state.user_location,
                        st.session_state.user_budget,
                    )
                with st.spinner("Creating the perfect answer for you...✨"):
                    audio_file = text_to_speech(final_response)
                    autoplay_audio(audio_file)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.session_state.response_generated = True  # Mark response as generated

                # Render assistant message and feedback immediately
                with st.chat_message("assistant"):
                    st.write(final_response)
                    feedback_key = f"feedback_{len(st.session_state.messages) - 1}"  # Unique key for feedback
                    feedback = streamlit_feedback(feedback_type="thumbs", align="flex-start", key=feedback_key)
                    if feedback:
                        st.session_state.feedbacks.append({
                            "message_id": len(st.session_state.messages) - 1,
                            "message": final_response,
                            "feedback": feedback
                        })
                os.remove(audio_file)

    # Float the footer container (place it at the bottom)
    footer_container.float("bottom: 0rem; right: 10px;")

########------------------------------

# Function to set background image using CSS
def set_background_image(image_url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Main App Logic
if not st.session_state.logged_in:
    # Set background image for the login page
    set_background_image('https://img.freepik.com/free-vector/marble-texture-background_125540-356.jpg?t=st=1731482349~exp=1731485949~hmac=d80f2d73266d5d927b2ce3246b8fb633b3f16c2d2ae370d7d3a0529bd11211d9')  # Replace with your image URL

    custom_login_css()
    st.markdown("<div style='text-align: center; margin: auto; padding: 2rem;'>", unsafe_allow_html=True)
    st.markdown("<h1> 𝚆𝚎𝚕𝚌𝚘𝚖𝚎 𝚝𝚘 𝚅𝚒𝚗𝚒!</h1>", unsafe_allow_html=True)
    st.session_state.name = st.text_input("Enter your Name", placeholder="Enter your name")
    st.session_state.number = st.text_input("Enter your Number", placeholder="Enter your mobile number")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Call login directly via on_click
    st.button("Login", on_click=login)
else:
    if st.button("🔙", key="logout_button", on_click=logout):
        logout()


    # Main Application Interface
    display_welcome_warning()
    voice_interface()