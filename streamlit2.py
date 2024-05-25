import streamlit as st
from transformers import pipeline

# Use Streamlit's caching to avoid reloading the model multiple times
@st.cache_resource
def load_model():
    model_name = "gpt-3.5-turbo"  # Replace with your model
    llm_pipeline = pipeline("text-generation", model=model_name)
    return llm_pipeline

# Function to generate response using the cached model
def generate_response(prompt):
    llm_pipeline = load_model()
    responses = llm_pipeline(prompt, max_length=100, num_return_sequences=1)
    return responses[0]['generated_text']

# Streamlit app
st.title("LLM Deployment with Streamlit")

st.write("Enter your prompt below:")

# Text input from the user
user_input = st.text_area("Prompt", "")

if st.button("Generate Response"):
    if user_input:
        # Generate response from the LLM
        response = generate_response(user_input)
        
        # Display the response
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter a prompt.")
