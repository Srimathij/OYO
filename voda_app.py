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
    if "welcome_played" not in st.session_state:
        st.session_state.welcome_played = False
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """
    ✨ **Hello and welcome!** ✨  
    I’m **Vini**, your personal Vodafone assistant, here to help you with transactions, exclusive offers, discounts, and services. Let's make your experience smooth and rewarding! 😊  

    💡 **Special Update:** Your current plan is about to expire, but no worries—I’ve got the perfect plans for you!  

    🎯 **Today’s Top Recommendation:**  
    - **Price:** ₹859    

    💥 **Special Deal:** Recharge ₹300 or more today and enjoy **10% cashback**!  

    What are you waiting for? Let me know how I can assist you further—let’s get started! 🎉
                """
            }
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

        1. Login Assistance:
            - Users log in by entering a username or phone number and password.
            - I securely validate the credentials and log the user in.
            - If login fails, I provide account recovery options, such as resetting the password, or share support contact details for assistance.

        2. Personalized Greetings:
            - After a successful login, I greet the user warmly by name.
            - I may acknowledge recent activities, such as "Your last recharge was ₹299 on [Date]."
            - I highlight special offers or promotions tailored to the user’s preferences or past interactions. For example:
                - "Today’s deal: Recharge with ₹300 or more and get 10% cashback!"

        3. Plan Status Check:
            - I automatically check the user’s current Vodafone recharge plan:
                - If the plan has expired, I initiate the Expired Plan Flow.
                - If the plan is active, I provide details, such as the expiry date, benefits, and potential Vodafone upgrades.
            - If the user asks, "What’s my current plan status?" respond with:
                - "Your current plan includes unlimited calls and 2GB/day, valid until Nov 29, 2024. Would you like to renew it or explore upgrades?"
            - If the user wants to check detailed account status, provide them with a link:
                - "You can check your account status and details by visiting [Vodafone Account Status Link]."
                - Replace `[Vodafone Account Status Link]` with the actual link to the user’s account page, such as a URL like `https://www.vodafone.in/my-account`.

        4. Recharge Plans:
            - I present a list of Vodafone-specific recharge plans customized to the user’s historical usage and preferences. For example:
                - Plan A: ₹199 | 1GB/day | 28 days.
                - Plan B: ₹299 | Unlimited calls + 2GB/day | 56 days.
            - Each plan includes cost, duration, and key benefits for easy comparison.

            - If the user asks questions like:
                - "What are the benefits of Plan A?"
                - "Can I switch plans mid-cycle?"
            I respond with real-time information to assist their decision-making.

        5. Plan Selection & Payment:
            - Once the user selects a plan, I confirm the selection with:
                - "Got it! Please give me a moment while I process your request..."
            - If the user says something like "I will go with UPI" or expresses intent to pay using UPI, I respond with:
                - "Please hold on while I process your transaction..."
                - [Wait for a minute as the transaction is processed...]
                - Once the payment is successful, I confirm the recharge with:
                    - "Success! Your new plan is now active."
            - I guide the user to a secure payment interface, offering options such as UPI, credit cards, or mobile wallets.
            - Upon successful payment, I confirm the recharge with:
                - "Success! Your new Vodafone plan is now active."
            - I update the user’s account instantly and notify them with details, such as:
                - "You can view the details of your active plan here."
                - "A confirmation message has been sent to your registered mobile number."

        6. Reminders & Notifications:
            - I notify users about the expiration of their plans with engaging messages, like:
                - "Your current plan will expire on [Date]. Would you like to upgrade or set a reminder?"
            - If the user agrees to a reminder, I schedule it and provide confirmation, such as:
                - "Reminder set! You’ll be notified a day before your plan expires."

        7. Suggestions & Upgrades:
            - Based on the user’s usage data, I suggest possible upgrades that may suit their needs better. For example:
                - "Upgrade to Plan C: Unlimited calls + 3GB/day for ₹399, valid for 84 days."
            - Each suggestion includes information on benefits, duration, and cost for easy comparison.

        8. Discounts & Offers:
            - If the user asks about Vodafone discounts or deals, I provide a carousel of exciting offers, each including:
                - Title: Offer name, such as "10% Cashback on Recharge."
                - Description: Key details, such as "Recharge with ₹300 or more to avail cashback."
                - Image: A visual representation of the deal.
                - Validity: The expiration date of the offer.

        9. Out-of-Scope Handling:
            - If a user asks a question unrelated to Vodafone products or services, I respond with:
                - "I’m here to assist with Vodafone-related queries only. For unrelated topics, I recommend consulting the relevant resources or support."

        10. Generalized Response:
            - Do not include any emoji characters.

        11. Response Length and Greeting Rules:
            - Do not generate lengthy responses. Keep answers concise and to the point.
            - Provide a greeting only in the welcome message. Avoid greetings in other responses.


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

def text_to_speech(input_text, slow=False, output_file="temp_audio_play.mp3"):
    """
    Converts text to speech using pyttsx3 with a female English voice.

    Args:
        input_text (str): The text to convert to speech.
        slow (bool): Whether the speech should be slower (default is False).
        output_file (str): The file path to save the audio (default is "temp_audio_play.mp3").

    Returns:
        str: The file path to the saved audio, or None if an error occurs.
    """
    try:
        # Initialize pyttsx3 engine
        engine = pyttsx3.init()

        # Adjust speech rate
        base_rate = 160  # Conversational rate
        engine.setProperty('rate', base_rate - 40 if slow else base_rate)

        # Adjust volume for clarity
        engine.setProperty('volume', 0.9)

        # Set female voice for English
        voices = engine.getProperty('voices')
        female_voice = None

        # Try to find a female English voice
        for voice in voices:
            if "female" in voice.id.lower():  # Look for "female" in voice ID
                female_voice = voice
                break
        
        # If a female voice is found, set it
        if female_voice:
            engine.setProperty('voice', female_voice.id)
            print(f"Using female voice: {female_voice.id}")
        else:
            # Fallback to the first voice if no female voice is found
            engine.setProperty('voice', voices[1].id)
            print(f"No female voice found. Using default voice: {voices[1].id}")

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
# Login functionality with loading buffer
def login():
    if st.session_state.name and st.session_state.number:
        # Show loading screen
        placeholder = st.empty()
        with placeholder.container():
            custom_loading_css()  # Add custom loading CSS
            st.markdown(
                """
                <div class='loading-container'>
                    <img src="https://upload.wikimedia.org/wikipedia/commons/7/72/Vodafone_Idea_logo.svg" class="logo-spin">
                    <h2 class="loading-text">Loading, please wait...</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(3)  # Simulate loading time
        placeholder.empty()

        # Log the user in
        st.session_state.logged_in = True
    else:
        st.error("Please enter both your name and number.")

# Login Page
# Login Page
def login_page():
    custom_css()

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/7/72/Vodafone_Idea_logo.svg",
        width=150,
        caption="Vodafone Idea"
    )
    st.markdown(
        "<div style='background-color: #ffcc00; padding: 10px; border-radius: 8px; color: black; font-weight: bold;'>5G for a better tomorrow</div>",
        unsafe_allow_html=True,
    )

    st.session_state.name = st.text_input("Enter your Name", placeholder="Enter your name")
    st.session_state.number = st.text_input("Enter your Number", placeholder="Enter your mobile number")
    st.button("Login", on_click=login)


# Toggle listening state
def toggle_listening():
    st.session_state.listening = not st.session_state.listening

def logout():
    st.session_state.logged_in = False
    st.session_state.welcome_played = False  # Reset welcome played status

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
###### Voice interface -------------------------

import streamlit as st
from streamlit_feedback import streamlit_feedback

def set_background_image(image_url: str):
    # Function to set a custom background image for the app
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

def voice_interface():
    # Set a custom background image
    set_background_image("https://png.pngtree.com/background/20210715/original/pngtree-white-simple-texture-background-picture-image_1323742.jpg")  # Replace with your image URL
    st.markdown(
        """
        <style>
        /* Example of custom styling for header and background */
        [data-testid="stHeader"] {
            background-image: linear-gradient(90deg, rgb(0, 102, 204), rgb(102, 255, 255));
        }
       
        </style>
        """,
        unsafe_allow_html=True
    )

    if "feedback_processed" not in st.session_state:
        st.session_state.feedback_processed = []

    st.markdown("<h4 style='text-align: center;'>ᴠᴏᴅᴀꜰᴏɴᴇ.ᴀɪ</h4>", unsafe_allow_html=True)

    # Footer for the audio recorder
    footer_container = st.container()
    with footer_container:
        audio_bytes = audio_recorder(
            text="𝙑𝙞𝙣𝙞 𝙝𝙚𝙧𝙚", 
            icon_name="person-dress",
            recording_color="#9e0d19",
            neutral_color="#9e0d19", 
            icon_size="3x",
            sample_rate=16000  
        )


    # Display all messages with feedback for assistant responses
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Add feedback widget for assistant messages
            if message["role"] == "assistant":
                feedback_key = f"feedback_{index}"  # Unique key for feedback widget
                feedback = streamlit_feedback(feedback_type="thumbs", align="flex-start", key=feedback_key)

                # Check if feedback has already been processed for this message
                if feedback and index not in st.session_state.feedback_processed:
                    st.session_state.feedback_processed.append(index)
                    st.session_state.feedbacks.append({
                        "message_id": index,
                        "message": message["content"],
                        "feedback": feedback
                    })

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
    footer_container.float("bottom: 0rem; right: 12px;")

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
    set_background_image('https://techcrunch.com/wp-content/uploads/2020/09/vi-vodafone-idea.jpg')  # Replace with your image URL

    custom_login_css()
    st.markdown("<div style='text-align: center; margin: auto; padding: 2rem;'>", unsafe_allow_html=True)
    st.markdown(
    "<h5 style='color: green;'>𝚅𝚒𝚗𝚒 𝙻𝚘𝚐𝚒𝚗</h5>",
    unsafe_allow_html=True
    )
    st.session_state.name = st.text_input("Enter your Name", placeholder="Enter your name")
    st.session_state.number = st.text_input("Enter your Number", placeholder="Enter your mobile number")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Call login directly via on_click
    st.button("Login", on_click=login)
else:
    # if st.button("🔙", key="logout_button", on_click=logout):
    #     logout()


    # Main Application Interface
    display_welcome_warning()
    voice_interface()