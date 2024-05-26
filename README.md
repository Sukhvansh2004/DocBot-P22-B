This repository comprises of Docbot which is a multimodal chatbot with current expanse to take in pdf inputs and then generating the answer and revelant image regarding the query. This is done in three phase:- 

# Pre-processing:- 

1. PDF Input goes in and with use of apache pdfbox library we parse it into text file
2. PDF file is also given into input to a function which we extract images and the revelant texts regarding it( above/below text of image)
3. The text file is sent through a sentence transformer which generates embedding space for the text 
4. The embedding file is then sent through ANNOY (Approximate Nearest Neighbour oh yeah) which makes tree structure correlating the embedding to each other 

# Query - Processing

1. Query comes in and is passed through Sentence Transformer and embeddings generated respectively.
2. Embeddings is then compared to pdf file's text embedding and then nearest k revelant sentence to query are retrieved (where k is hyperparameter)
3. The Query and nearest k sentence are sent through LLM Model (gemma 2b in our case) and revelant output is generated 
4. The output embedding representation is compared to the image captions and most revelant (by use of annoy algorithm) is then selected. 

# Output: 
The answer and revelant image are given as output. 


We support japanese language aswell as of right now and it basically same but the embedding generation is done using bert trained on japanese wiki 

# TO-DO 
1. Deploy the site and support multiple input type eg. docx
2. Better database system 
3. Voice query support and output TTS

Details regarding the project and model:
    The details of the project and the model are given in the power point presentation named 'DOCBOT.pptx' in the current directory. A working vedio of the model name 'Working Clip.mp4' is also provided in the current directory.

# Prerequisites:

Python 3.10 or later
pip package installer
PyTorch (with CUDA support) if using Windows (installation instructions on PyTorch's official website)
Installation:

```shell
#Clone the repository:

git clone https://github.com/Sukhvansh2004/DocBot-P22-B.git

#Navigate to the cloned repository directory and install the required packages:

pip install -r requirements.txt

#Running the Application:
#Ensure you have the necessary prerequisites installed. See Important Notes
#Start the Streamlit app:

streamlit run streamlit.py
```

# Important Notes:

Remember to install PyTorch with CUDA support on your device as per the official PyTorch installation instructions.

Additional Considerations:
For enhanced performance and scalability, consider using a GPU-accelerated environment.
Explore advanced question-answering techniques to improve the chatbot's accuracy and versatility.