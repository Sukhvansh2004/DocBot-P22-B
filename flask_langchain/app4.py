from flask import Flask, request, send_from_directory, Response, jsonify
import pdfbox
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import torch

app = Flask(__name__)

datafolder = 'data'
os.makedirs(datafolder, exist_ok=True)  # Create the 'data' folder if it doesn't exist

auth_token = os.environ.get("HF_TOKEN")  # was a hard-coded token; export HF_TOKEN instead

sbert_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1', use_auth_token=auth_token)  # Example model, replace with your desired SBERT model

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16
)
generation_params = {
    "max_length":1000
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Move SBERT model to CUDA if available
sbert_model = sbert_model.to(device)

# Move LM model to CUDA if available
model = model.to(device)

# Ensure that the LM model is in evaluation mode
model.eval()



def pdf2text(pdf):
    p = pdfbox.PDFBox()
    p.extract_text(pdf, os.path.join(datafolder, 'output.txt'))
    delete_ann_file()
    print('converted')

def text2embedd2query(text_file, query):
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
    annoy_index_path = os.path.join(datafolder, 'index.ann')
    annoy_index = load_annoy_index_from_file(vector_dimension, annoy_index_path)
    if annoy_index is None:
        annoy_index = AnnoyIndex(vector_dimension, 'angular')
        for i, vector in enumerate(paragraph_embeddings):
            annoy_index.add_item(i, vector)
        annoy_index.build(n_trees=15)
        save_annoy_index_to_file(annoy_index, annoy_index_path)

    num_neighbors = 5
    query_embedding = sbert_model.encode([query])[0]
    num_neighbors = 5

    nearest_neighbor_indices = annoy_index.get_nns_by_vector(query_embedding, num_neighbors)

    nearest_neighbor_paragraphs = [paragraphs[index] for index in nearest_neighbor_indices]
    return nearest_neighbor_paragraphs
 
def save_annoy_index_to_file(annoy_index: AnnoyIndex, file_path):
    annoy_index.save(file_path)

def load_annoy_index_from_file(vector_dimension, file_path):
    if os.path.exists(file_path):
        annoy_index = AnnoyIndex(vector_dimension, 'angular')
        annoy_index.load(file_path)
        return annoy_index
    else:
        return None

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
    neighbor = text2embedd2query(os.path.join(datafolder, 'output.txt'), query)
    combined_string = ' '.join(neighbor)
    input_text = tokenizer("This is my query: " + '\n' + query + ' and this is the context: '+ '\n' + combined_string + '\n' + " Answer the query accordingly ",return_tensors="pt").to(device="cuda")

    outputs = model.generate(**input_text,**generation_params)
    
    output = tokenizer.decode(outputs[0])
    
    def generate_response(output: str):
        output = output[output.find("Answer the query accordingly")+len("Answer the query accordingly"):]
        yield output.encode('utf-8')

    return Response(generate_response(output), mimetype='text/html')

@app.route('/delete_ann_file', methods=['POST'])
def delete_ann_file():
    annoy_index_path = os.path.join(datafolder, 'index.ann')
    if os.path.exists(annoy_index_path):
        os.remove(annoy_index_path)
        print(f"Deleted {annoy_index_path}")
    return "OK"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500)