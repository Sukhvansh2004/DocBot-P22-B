DocBot-P22-B: A Chatbot for PDF Information Extraction and Question 

This project presents DocBot-P22-B, a user-friendly chatbot built using Google Gemma and Streamlit for extracting information from PDFs and answering your questions related to the content.

Key Features:

1. PDF Processing:
Leverages Google Gemma's capabilities to extract text and relevant information (along with images) from PDFs.
2. Provides a user interface for uploading and processing PDFs.
3. Question Answering:
Enables you to ask questions about the extracted information from the PDF.
Employs a robust question-answering system to deliver accurate responses.
4. Streamlit Deployment:
Streamlit facilitates a user-friendly web application for easy interaction with DocBot-P22-B.
5. Multilingual: The model can be used for both English and Japanese texts, given the input pdfs are also in those languages respectively.

Prerequisites:

Python 3.10 or later
pip package installer
PyTorch (with CUDA support) if using Windows (installation instructions on PyTorch's official website)
Installation:

Clone the repository:

Bash
git clone https://github.com/Sukhvansh2004/DocBot-P22-B.git
Use code with caution.
content_copy
Navigate to the cloned repository directory and install the required packages:

Bash
pip install -r requirements.txt

Running the Application:

Ensure you have the necessary prerequisites installed.

Start the Streamlit app:

Bash
streamlit run streamlit.py

Important Notes:

The data folder is to be deleted when running the streamlit run streamlit.py command to prevent data conflicts.
Remember to install PyTorch with CUDA support on your device as per the official PyTorch installation instructions.
Additional Considerations:

For enhanced performance and scalability, consider using a GPU-accelerated environment.
Explore advanced question-answering techniques to improve the chatbot's accuracy and versatility.