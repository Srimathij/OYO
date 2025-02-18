import streamlit as st
import time

# Initialize session state
def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "✨ Hey there! I’m Vini, your Vodafone assistant, here to help with transactions, discounts, offers, and services—just ask! ✨"
            }
        ]
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "feedbacks" not in st.session_state:
        st.session_state.feedbacks = []
    if "name" not in st.session_state:
        st.session_state.name = ""
    if "number" not in st.session_state:
        st.session_state.number = ""
    if "listening" not in st.session_state:
        st.session_state.listening = False

initialize_session_state()

# Custom CSS for styling (responsive)
def custom_css():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #121212; /* Dark background */
            color: white;
        }
        .stApp {
            background-color: #121212;
        }
        .stTextInput input {
            border: 2px solid #ff0000; /* Red border for inputs */
            border-radius: 8px;
            padding: 10px;
            background-color: #1e1e1e; /* Darker background for inputs */
            color: white;
        }
        .stButton button {
            background-color: #ff0000; /* VI red */
            color: white;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            margin-top: 10px;
            cursor: pointer;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease-in-out;
        }
        .stButton button:hover {
            background-color: #e60000; /* Slightly darker red on hover */
            transform: scale(1.05);
        }
        .assistant-message {
            color: #FFD700; /* Golden Yellow */
        }
        .logout-container {
            display: flex;
            justify-content: flex-end;
        }
        .chat-container {
            max-width: 700px;
            margin: auto; /* Center align chat container */
        }
        .message-container {
            padding: 10px;
            background-color: #1e1e1e;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        @media (min-width: 768px) {
            .chat-container {
                max-width: 80%;
            }
            .stTextInput input {
                font-size: 1.1rem; /* Larger text for desktop */
            }
            .stButton button {
                padding: 12px 25px; /* Bigger buttons for desktop */
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Custom CSS for loading screen with rotating logo
def custom_loading_css():
    st.markdown(
        """
        <style>
        .loading-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #ffcc00; /* Golden yellow background */
        }
        .logo-spin {
            width: 150px;
            animation: spin 2s infinite linear;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        .loading-text {
            margin-top: 20px;
            font-size: 1.5rem;
            color: #121212; /* Dark text */
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Logout functionality
def logout():
    st.session_state.logged_in = False
    st.session_state.name = ""
    st.session_state.number = ""

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

# Chatbot interface
def chatbot_interface():
    custom_css()

    # Logo and slogan
    st.image(
        "https://zerovalueinfo.wordpress.com/wp-content/uploads/2020/09/vodafone-idea-new-logo-vi-1200.jpg",
        use_container_width=True,
    )
    st.markdown(
        "<div style='background-color: #ff0000; padding: 15px; font-size: 1.5rem; color: white; text-align: center;'>5G for a better tomorrow</div>",
        unsafe_allow_html=True,
    )

    # Logout button
    st.markdown("<div class='logout-container'>", unsafe_allow_html=True)
    if st.button("Logout"):
        logout()
    st.markdown("</div>", unsafe_allow_html=True)

    # Chatbox area
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("<div class='message-container'>", unsafe_allow_html=True)

    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: right; color: black;'>{message['content']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Form for user input
    with st.form(key="chat_form"):
        user_input = st.text_input("Your message:", placeholder="Type your message here...", key="user_input")
        submit = st.form_submit_button("Submit")

    if submit and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = f"This is a placeholder response to '{user_input}'."
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Microphone button outside the form
    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
    with col2:
        if st.button("🎤 Click to Speak"):
            toggle_listening()

    # Listening popup
    if st.session_state.listening:
        st.markdown(
            "<div style='text-align: center; background-color: #ffcc00; padding: 10px; margin-top: 10px; border-radius: 8px;'>🎤 Listening...</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# Main Application Logic
if not st.session_state.logged_in:
    login_page()
else:
    chatbot_interface()
