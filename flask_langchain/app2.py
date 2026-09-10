# from langchain.llms import CTransformers
from flask import Flask, request, send_from_directory
import pdfbox
from flask import Response
import os
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import jsonify
import transformers
import torch 



auth_token = os.environ.get("HF_TOKEN")  # was a hard-coded token; export HF_TOKEN instead

sbert_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1',use_auth_token=auth_token)  # Example model, replace with your desired SBERT model

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

app = Flask(__name__)
datafolder = 'data'
os.makedirs(datafolder, exist_ok=True)  # Create the 'data' folder if it doesn't exist

def pdf2text(pdf):
    p = pdfbox.PDFBox()
    p.extract_text(pdf,r'data\output.txt') 
    print('converted')

def text2embedd2query(text_file,query):
    paragraphs = []
    with open(text_file, 'r', encoding='utf-8') as file:
        current_paragraph = ""
        for line in file:
            current_paragraph += line.strip() + " "
            if '.' in line:
                if current_paragraph.count('.') >= 5: 
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
    paragraph_embeddings = [np.array(sbert_model.encode([paragraph])[0]) for paragraph in paragraphs]

    vector_dimension = len(paragraph_embeddings[0])
    annoy_index = AnnoyIndex(vector_dimension, 'angular')  
    for i, vector in enumerate(paragraph_embeddings):
        annoy_index.add_item(i, vector)
    annoy_index.build(n_trees=15)
    num_neighbors = 5
    query_embedding = sbert_model.encode([query])[0]
    num_neighbors = 5

    nearest_neighbor_indices = annoy_index.get_nns_by_vector(query_embedding, num_neighbors)

    nearest_neighbor_paragraphs = [paragraphs[index] for index in nearest_neighbor_indices]
    return  nearest_neighbor_paragraphs


@app.route('/get_initial_pdf')
def get_initial_pdf():
    # Replace this with the path to your initial PDF file
    initial_pdf_path = os.path.join(datafolder, 'initial.pdf')

    with open(initial_pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    return jsonify({'pdf_bytes': pdf_bytes.hex()})



@app.route('/')
def chatbot():
    return send_from_directory('', 'chatbot2.html')


@app.route('/refresh')
def refresh():
    # TODO
    print("Data refreshed")
    return "Data refreshed"

@app.route('/upload', methods=['POST'])
def upload_pdf():
    uploaded_file = request.files['pdfFile']
    if uploaded_file:
        file_path = os.path.join(datafolder, uploaded_file.filename)
        uploaded_file.save(file_path)
        print(f"PDF file '{uploaded_file.filename}' saved to '{file_path}'")
        pdf2text(file_path)

        return f"PDF file '{uploaded_file.filename}' uploaded and processed successfully."
    else:
        return "No PDF file received."

@app.route('/send')
def getresponse():
    query = request.args.get('query', '')
    language = request.args.get('language', 'en')
    print(f"Query Received: {query} (Language: {language})")
    neigbhour = text2embedd2query(r'data/output.txt',query)
    combined_string = ' '.join(neigbhour)
    output = pipeline("This is my query: " + '\n' + query + ' and this is the context: '+ '/n' + combined_string + '\n' + " Answer the query accordingly") 
    
    def generate_response():
        yield output

    return Response(generate_response(), mimetype='text/html')
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500)