from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from flask import Flask, request, send_from_directory
import base64

from flask import Response
import time
import os
import glob

app = Flask(__name__)
datafolder = 'data'
os.makedirs(datafolder, exist_ok=True)  # Create the 'data' folder if it doesn't exist





def upload_images(page_no):
  print("ENtered---------------------------------------------")
  print("page NO------>", page_no)
  """
  Uploads all images in a directory whose filename contains a specific pattern.

  Args:
      upload_function (function): A function that takes an image path as input and performs the upload.
      directory (str, optional): The directory containing the images. Defaults to current directory (".).
  """
  directory="./images"
  dir = ""
  # Iterate through directory entries
  for entry in os.scandir(directory):
    if entry.is_file() and entry.name.endswith(".jpg") and f"extracted_image_file_1_{page_no}_" in entry.name:
      image_path = os.path.join(directory, entry.name)
      dir = image_path

  print("dir",dir)
  return dir


def list_pdf_files_glob():
    # Construct the search pattern
    search_pattern = os.path.join(datafolder, '**', '*.pdf')
    # Use glob to find all PDF files in the directory and subdirectories
    pdf_files = glob.glob(search_pattern, recursive=True)
    # print(pdf_files)
    return pdf_files

FILE_LOADER_MAPPING = {
    "pdf": (PyMuPDFLoader, {}),
}

config = {
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.8,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'gpu_layers': 1  # 'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    **config
)


model_name = "All-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'} #mac M1 have mps (check system config on your system)
#model_kwargs = {'device': 'cpu'} - its generic for CPU on any machine but is damn slow. u can use cuda on intel/nvidia setup but change the param
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loaded_documents = []
loaders = [PyMuPDFLoader(x) for x in list_pdf_files_glob()]
for loader in loaders:
    loaded_documents.extend(loader.load())

#loaded_documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30, length_function = len)
chunked_documents = text_splitter.split_documents(loaded_documents)
persist_directory = 'db'
db = FAISS.from_documents(chunked_documents, embeddings)

retriever = db.as_retriever(search_kwargs={"k":1})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, verbose=True)

app = Flask(__name__)


def load_japanese_models():
    japanese_model_name = "your_japanese_model_name"
    japanese_model_kwargs = {'device': 'cuda'}  # or 'device': 'cpu' if no GPU available
    japanese_encode_kwargs = {'normalize_embeddings': True}
    japanese_embeddings = HuggingFaceBgeEmbeddings(
        model_name=japanese_model_name,
        model_kwargs=japanese_model_kwargs,
        encode_kwargs=japanese_encode_kwargs
    )

    japanese_llm = CTransformers(
        model="your_japanese_model_path",
        model_file="your_japanese_model_file.gguf",
        model_type="mistral",
        **config
    )

    return japanese_llm, japanese_embeddings

@app.route('/send')
def getresponse():
    query = request.args.get('query', '')
    language = request.args.get('language', 'en')
    print(f"Query Received: {query} (Language: {language})")

    if language == 'ja':
        japanese_llm, japanese_embeddings = load_japanese_models()
        japanese_db = FAISS.from_documents(chunked_documents, japanese_embeddings)
        japanese_retriever = japanese_db.as_retriever(search_kwargs={"k": 1})
        qa_model = RetrievalQA.from_chain_type(llm=japanese_llm, chain_type="stuff", retriever=japanese_retriever, return_source_documents=True, verbose=True)
        response = qa_model(query)
    else:
        response = qa(query)

    print(response)
    lines = response['result'].split('\n')
    relevant_page = response['source_documents'][0].metadata['page']  # Get the relevant page number
    image_path = upload_images(relevant_page)  # Construct the image path based on the page number
    print("IMAGE PATH:", image_path)

    # def generate_response():
    #     for line in lines:
    #         yield line + "\n"
    #     if os.path.exists(image_path):
    #         print("entered-----------------------------------------------")
    #         yield f"<img src='{image_path}' alt='Retrieved Image'>"
    
    def generate_response():
        for line in lines:
            yield line + "\n"
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image_data = f.read()
            # yield f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"
            yield f"{image_path}"

    return Response(generate_response(), mimetype='text/html')








@app.route('/')
def chatbot():
    return send_from_directory('frontend', 'chatbot.html')

# @app.route('/send')
# def getresponse():
#     query = request.args.get('query', '')
#     print("Query Received: ", query)
#     response = qa(query)
#     print(response)
#     lines = response['result'].split('\n')
#     print(lines)
#     return f"{lines}"

# @app.route('/send')
# def getresponse():
#     query = request.args.get('query', '')
#     print("Query Received: ", query)

#     def generate_response():
#         # Simulate a long-running task
#         total_steps = 100
#         for i in range(total_steps):
#             # Send progress update to the client
#             progress = f"Progress: {i}/{total_steps}\n"
#             yield progress

#             # Simulate some work
#             time.sleep(0.3)

#         response = qa(query)
#         print(response)
#         lines = response['result'].split('\n')
#         print(lines)

#         # Send the final result
#         for line in lines:
#             yield line + "\n"

#     return Response(generate_response(), mimetype='text/plain')


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

        # After saving the PDF, call the LLM model
        # Load the new PDF file
        loader = PyMuPDFLoader(file_path)
        new_documents = loader.load()

        # Split the new documents into chunks
        new_chunked_documents = text_splitter.split_documents(new_documents)

        # Add the new chunks to the database
        db.add_documents(new_chunked_documents)

        return f"PDF file '{uploaded_file.filename}' uploaded and processed successfully."
    else:
        return "No PDF file received."

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500)