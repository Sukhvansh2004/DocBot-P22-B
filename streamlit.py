import base64
import streamlit as st
import random
import time
import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate
from streamlit_authenticator.utilities.hasher import Hasher
import yaml
from yaml.loader import SafeLoader
from Backend import text_gen  # Import the TextGen class from Backend
import os
from Backend2 import main_current
from PIL import Image
# Function to display PDF from bytes
def display_pdf_from_bytes(pdf_data):
    pdf_display = (
        f'<embed src="data:application/pdf;base64,{pdf_data}" '
        'width="500" height="800" type="application/pdf"></embed>'
    )
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

def toggle_related_images():
    if st.session_state.show_related_images:
        st.session_state.show_related_images = False
        return "View related images"
    else:
        st.session_state.show_related_images = True
        return "Hide images"

# Function to display related images in a carousel layout
def display_related_images():
    image_paths = ["image1.jpg", "image2.jpg"]  # Update these paths to the locations of your images
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        st.sidebar.image(image_bytes, caption="Related Image", use_column_width=True)

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
str2=[0]
# Main function
def main():
    st.set_page_config(page_title="DocBot", page_icon="🛵", layout="wide", initial_sidebar_state="expanded")

    passwords_to_hash = ['abc', 'def']
    hashed_passwords = Hasher(passwords=passwords_to_hash).generate()

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    _, authentication_status, _ = authenticator.login(location='main')


    if authentication_status:
        st.sidebar.markdown("# © DocBot; developed by P22-B")
        st.sidebar.image("your_logo.png", width=250)
        st.sidebar.title("PDF Viewer")
        st.title("Welcome to ドキュメントボット! 🛵🤖")  # Add the header here

        # Language selection dropdown
        selected_language = st.sidebar.selectbox("Select Language", options=["English", "Japanese"], index=0)
        print(selected_language)
        st.session_state.language = selected_language

        typing_text = st.empty()
        message = "Hello there! How can I assist you today?"

        typed_message = ""
        for char in message:
            typed_message += char
            # Apply white color to the typed message
            typing_text.markdown(
                f'### <div style="color: red;">{typed_message}</div>',
                unsafe_allow_html=True)
            time.sleep(0.05)  # Adjust typing speed (seconds)

        # Initialize session state
        # if "show_related_images" not in st.session_state:
        #     st.session_state.show_related_images = False

        # Streamlit layout setup
        # st.sidebar.image("your_logo.png", width=300)  # Replace "your_logo.png" with the path to your logo
        # st.sidebar.markdown("<h1 style='text-align: center; color: grey;'>Welcome to DocBot!</h1>", unsafe_allow_html=True)  # Add your text here


        # PDF uploader
        uploaded = st.sidebar.file_uploader(label="Please browse for a PDF file", type="pdf")
        if uploaded is not None:
            print(uploaded)
            # Display PDF directly from bytes
            # Process the PDF using TextGen class
            str1=[uploaded.name+" "+selected_language]
            if str1[0]!=str2[0]:
                text_gen.process_pdf(uploaded,selected_language)
                # Display PDF on the sidebar
                base64_pdf = base64.b64encode(uploaded.read()).decode("utf-8")
                display_pdf_from_bytes(base64_pdf)
                str2[0]=str1[0]


        # Clickable element to view or hide related images
        # toggle_button_label = toggle_related_images()
        # if st.sidebar.button(toggle_button_label):
        #     toggle_related_images()

        # # Display or hide related images based on toggle state
        # if st.session_state.show_related_images:
        #     display_related_images()

        # authenticator.logout('Logout', 'sidebar')

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            str1=[uploaded.name+" "+selected_language]
            if str1[0]!=str2[0]:
                text_gen.process_pdf(uploaded,selected_language)
                # Display PDF on the sidebar
                base64_pdf = base64.b64encode(uploaded.read()).decode("utf-8")
                display_pdf_from_bytes(base64_pdf)
                str2[0]=str1[0]
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            response = {}
            # Process the query using TextGen class
            response['text'] = text_gen.process_query(prompt,selected_language)
            response['image'] = main_current(prompt, os.path.join("data", uploaded.name), selected_language)
            import pandas as pd
            pd.DataFrame(response).to_csv("Tezt.csv")
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response['text'].decode('utf-8'))
                st.session_state.messages.append({"role":"assistant","content":response['text'].decode('utf-8')})
            for image_path in response['image']:    
                st.image(image_path)
                st.session_state.messages.append({"role":"assistant","content":Image.open(image_path)})

if __name__ == "__main__":
    main()