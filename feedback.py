import streamlit as st
from streamlit_feedback import streamlit_feedback

def handle_feedback():
    """
    Handles feedback submission when the user clicks 'Save feedback'.
    """
    feedback_key = st.session_state.get("fb_k", None)
    if feedback_key:
        st.session_state.feedback_processed.append(feedback_key)  # Mark feedback as processed
        st.toast("✔️ Feedback received!")
        st.write(f"Feedback saved for key: {feedback_key}")


def voice_interface():
    # Initialize session state variables
    if "feedback_processed" not in st.session_state:
        st.session_state.feedback_processed = []  # Tracks which messages have feedback submitted
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Stores chat messages

    st.markdown("<h3 style='text-align: center;'>𝐕𝐎𝐃𝐀𝐅𝐎𝐍𝐄.𝐀𝐈</h3>", unsafe_allow_html=True)

    # Footer for the audio recorder
    footer_container = st.container()
    with footer_container:
        audio_bytes = audio_recorder(text=None, icon_size="3x", sample_rate=16000)  # Replace with actual audio_recorder logic if required

    # Display all messages and feedback forms
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Feedback form for assistant messages
            if message["role"] == "assistant" and index not in st.session_state.feedback_processed:
                with st.form(f"feedback_form_{index}"):
                    streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="Enter your feedback here",
                        align="flex-start",
                        key=f"fb_k_{index}"  # Unique key for each feedback form
                    )
                    st.form_submit_button("Save feedback", on_click=handle_feedback)

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
                response = "😊"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.response_generated = True  # Mark response as generated

                # Render assistant message and feedback immediately
                display_carousel(offers_data)

            elif is_off_query(st.session_state.user_query):
                # Handle off-related queries
                response = "😊"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.response_generated = True  # Mark response as generated

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

                # Render assistant message
                with st.chat_message("assistant"):
                    st.write(final_response)
                os.remove(audio_file)

    # Float the footer container (place it at the bottom)
    footer_container.float("bottom: 0rem; right: 10px;")




    ########


    # Generate assistant response only if not already generated
if st.session_state.user_query and not st.session_state.response_generated:
    # Avoid response generation if feedback is already processed
    if st.session_state.user_query not in st.session_state.feedback_processed:
        if is_offer_query(st.session_state.user_query):
            # Handle offer-related queries
            response = "😊"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.response_generated = True  # Mark response as generated

            # Render assistant message and feedback immediately
            with st.chat_message("assistant"):
                st.write(response)

            # Immediate feedback form
            with st.form(f"feedback_form_{len(st.session_state.messages) - 1}"):
                streamlit_feedback(
                    feedback_type="thumbs",
                    optional_text_label="Enter your feedback here",
                    align="flex-start",
                    key=f"fb_k_{len(st.session_state.messages) - 1}"  # Unique key
                )
                st.form_submit_button("Save feedback", on_click=handle_feedback)

            display_carousel(offers_data)

        elif is_off_query(st.session_state.user_query):
            # Handle off-related queries
            response = "😊"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.response_generated = True  # Mark response as generated

            # Render assistant message and feedback immediately
            with st.chat_message("assistant"):
                st.write(response)

            # Immediate feedback form
            with st.form(f"feedback_form_{len(st.session_state.messages) - 1}"):
                streamlit_feedback(
                    feedback_type="thumbs",
                    optional_text_label="Enter your feedback here",
                    align="flex-start",
                    key=f"fb_k_{len(st.session_state.messages) - 1}"  # Unique key
                )
                st.form_submit_button("Save feedback", on_click=handle_feedback)

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

            # Immediate feedback form
            with st.form(f"feedback_form_{len(st.session_state.messages) - 1}"):
                streamlit_feedback(
                    feedback_type="thumbs",
                    optional_text_label="Enter your feedback here",
                    align="flex-start",
                    key=f"fb_k_{len(st.session_state.messages) - 1}"  # Unique key
                )
                st.form_submit_button("Save feedback", on_click=handle_feedback)

            os.remove(audio_file)
