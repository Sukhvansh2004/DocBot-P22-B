import streamlit as st
import time
import base64

# Function to simulate data loading or computation
def load_data():
    # Simulate data loading by sleeping for a few seconds
    time.sleep(3)
    return "Data loaded successfully!"

# Main function
def main():
    st.title("Streamlit Custom Loading Animation Demo")

    # Add a button to trigger data loading
    if st.button("Load Data"):
        file_ = open("loading.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        # Define text to display next to the GIF
        text = "Loading data..."

        # Adjust the width of the GIF
        gif_width = 50  # Set the desired width here

        # Set the font size of the text
        text_font_size = "20px"  # Set the desired font size here

        # Create a placeholder for the loading animation
        loading_animation = st.markdown(
            f'<div style="display: flex; align-items: center;"><img src="data:image/gif;base64,{data_url}" alt="loading gif" style="width: {gif_width}px;"><span style="font-size: {text_font_size}; margin-left: 10px;">{text}</span></div>',
            unsafe_allow_html=True,
        )

        # Simulate data loading
        data = load_data()

        # Remove the loading animation
        loading_animation.empty()

        # Display success message
        st.success(data)

if __name__ == "__main__":
    main()