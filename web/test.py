from flask import Flask, request, send_from_directory, Response, jsonify
import pdfbox
from base64 import b64encode
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer , util
import torch
import pickle
import fitz
# import util\
import PIL

from PIL import Image
# import pymupdf
import pikepdf
import io
import glob


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


datafolder = 'data'

Current_file = "data/file_1_unlocked.pdf"

# @app.route('/delete_ann_file', methods=['POST'])
# def delete_ann_file():
#     annoy_index_path = os.path.join(datafolder, 'index.ann')
#     if os.path.exists(annoy_index_path):
#         os.remove(annoy_index_path)
#         print(f"Deleted {annoy_index_path}")
#     return "OK"
def pdf2text(pdf):
    p = pdfbox.PDFBox()
    p.extract_text(pdf, os.path.join(datafolder, 'output.txt'))
    delete_ann_file()
    print('converted')
    return p

def text2embedd2query(query, pdf_path):
    with open(f"{pdf_path}_prev_text.pickle", "rb") as f:
        retrieved_list = pickle.load(f)
    sentences = retrieved_list
    sentences.reverse()
    paragraphs = [sentence for sentence in sentences]
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
    
    # nearest_neighbor_paragraphs = " Hello "
    return nearest_neighbor_paragraphs

def find_most_similar_sentence(sentences, xrefs, index, image_xref):
    list_hold = [image_xref]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    top = index + 5
    while index < top:
        try:
            embedding_1 = model.encode(sentences[index], convert_to_tensor=True)
            embedding_2 = model.encode(sentences[index + 1], convert_to_tensor=True)
            val = util.pytorch_cos_sim(embedding_1, embedding_2)
            value = val.item()
            print("val", value)

            if value >= 0.93:
                list_hold.append(xrefs[index])
            else:
                break
            index += 1
        except:
            index +=1
            print("none")

    return list_hold

def display_images(pdf_name, pair):
    for i in pair:
        img = Image.open(f'extracted_image_{pdf_name}_{i}.jpg')
        from IPython.display import display
        display(img)


def remove_non_utf8(text):
  """
  This function removes all characters from a string that are not valid UTF-8 encoded.

  Args:
      text: The string to be processed.

  Returns:
      A string containing only valid UTF-8 characters with replacements for invalid characters.
  """
  try:
    return text.encode('utf-8').decode('utf-8')
  except UnicodeDecodeError:
    # Replace non-utf8 characters with '?'
    return ''.join(char if ord(char) < 128 else ' ' for char in text)

# Example usage

def extract_text_and_images(pdf_path):
    text_blocks = []
    images = {}
    xref_page = {}
    text_xref_pairs = {}
    text_xref_pairs_list = []
    text_xref_pairs_list_rev = {}

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        i = 0
        page_data = ""

        while i < len(page.get_text("blocks")) and i < len(page.get_images()):
            # Extract text block
            if i < len(page.get_text("blocks")):
                block = page.get_text("blocks")[i]
                text = block[4]
                x0, y0, x1, y1 = block[:4]
                # text = "This is some text with non-utf8 characters like ❤."
                text = remove_non_utf8(text)

                text_blocks.append(text)
                text = str(text)
                page_data += str(text)
                lines = text.splitlines()
                try:
                    last_line = lines[-2] + " " + lines[-1]
                except:
                    last_line = lines[-1]
                prev_text = last_line

            # Extract image
            if i < len(page.get_images()):
                imglist = page.get_images()[i]
                xref = imglist[0]
                img_data = doc.extract_image(xref)
                images[xref] = img_data
                xref_page[xref] = page_num
                text_xref_pairs[xref] = prev_text
                text_xref_pairs_list.append([xref, prev_text])
                text_xref_pairs_list_rev[prev_text] = xref
                prev_text = None

            i += 1

    doc.close()
    return text_blocks, images, xref_page, text_xref_pairs, text_xref_pairs_list, text_xref_pairs_list_rev

def process_images(pdf_path, pdf_name, images, xref_page, text_xref_pairs, text_xref_pairs_list):
    text_xref_pairs_1 = {}
    text_xref_pairs_list_1 = []
    text_list_1 = []
    print("enter")
    for xref, image_data in images.items():
        print("plz")
        size = int(image_data["width"]) * int(image_data["height"])
        image_data["cs-name"] = "DeviceGray"
        image_data["colorspace"] = 1
        image_data["ext"] = "jpeg"
        try:
            img = Image.open(io.BytesIO(image_data["image"]))
        except:
            continue
        img = img.convert('RGB')

        if image_data["width"] > 150 and image_data["height"] > 150:
            text_xref_pairs_1[xref] = text_xref_pairs[xref]
            prev_text = text_xref_pairs[xref]
            text_xref_pairs_list_1.append([xref, prev_text])
            text_list_1.append(prev_text)

            filename = f"images/extracted_image_{pdf_name}_{xref}.jpg"
            try:
                img.save(filename)
            except:
                print("Error")

    return text_xref_pairs_1, text_xref_pairs_list_1, text_list_1

def save_data(pdf_path, text_xref_pairs_list_1, text_list_1):
    with open(f"{pdf_path}_text_xref_pairs.pickle", "wb") as f:
        pickle.dump(text_xref_pairs_list_1, f)

    with open(f"{pdf_path}_prev_text.pickle", "wb") as f:
        pickle.dump(text_list_1, f)

def unlock_pdf(pdf_path):
    filename = pdf_path

    with pikepdf.open(filename, allow_overwriting_input=True) as pdf:
        print(f"File {filename} opened correctly.")
        pdf.save(filename)
    print(f"File {filename} unlocked.")

def prev_main():
    # pdf_files = list_pdf_files_glob(datafolder)

    # for pdf_path in pdf_files:
        global Current_file 
        pdf_path = Current_file
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        unlock_pdf(pdf_path)

        text_blocks, images, xref_page, text_xref_pairs, text_xref_pairs_list, text_xref_pairs_list_rev = extract_text_and_images(pdf_path)
        print("len = ",len(text_xref_pairs_list))

        # text_blocks = pdf2text(Current_file)
        
        text_xref_pairs_1, text_xref_pairs_list_1, text_list_1 = process_images(pdf_path, pdf_name, images, xref_page, text_xref_pairs, text_xref_pairs_list)

        save_data(pdf_path, text_xref_pairs_list_1, text_list_1)

        # text = ""
        # for i in text_blocks:
        #     # print(i)
        #     text += str(i)
        # with open("data/output.txt", "w") as file:
        #     # Write the text to the file
        #     file.write(text)
            # .encode('utf-8')
        
        print(f"Extracted text blocks and {len(images)} images.")

def main_current(query):
    # pdf_files = list_pdf_files_glob(datafolder)

    # for pdf_path in pdf_files:
        global Current_file 
        pdf_path = Current_file
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # query = "Instrument and  control functions  4 - 3 3s"
        nearest = text2embedd2query(query, pdf_path)
        nearest_top = nearest[0]

        with open(f"{pdf_path}_text_xref_pairs.pickle", "rb") as f:
            retrieved = pickle.load(f)

        retrieved_list = []
        index = 0
        for i in retrieved:
            retrieved_list.append(i[1])
            if nearest[0] == i[1]:
                image_xref = i[0]
                break
            index += 1

        print(image_xref)

        with open(f"{pdf_path}_prev_text.pickle", "rb") as f:
            retrieved_list = pickle.load(f)

        with open(f"{pdf_path}_text_xref_pairs.pickle", "rb") as f:
            list2 = pickle.load(f)

        xrefs = []
        for i in list2:
            xrefs.append(i[0])

        print("len = ",len(retrieved_list))

        pair = find_most_similar_sentence(retrieved_list, xrefs, index, image_xref)
        print(pair)

        # display_images(pdf_name, pair)

prev_main()



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
    search_pattern = os.path.join(datafolder, '*', '.pdf')
    # Use glob to find all PDF files in the directory and subdirectories
    pdf_files = glob.glob(search_pattern, recursive=True)
    # print(pdf_files)
    return pdf_files

def text2embedd2query2(text_file, query):
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
    
    nearest_neighbor_paragraphs =  []
    for i in nearest_neighbor_indices:
        try:
            nearest_neighbor_paragraphs.append(paragraphs[i])  
        except:
            continue
    # nearest_neighbor_paragraphs = [print(paragraphs[index]) for index in nearest_neighbor_indices]
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

@app.route('/delete_ann_file', methods=['POST'])
def delete_ann_file():
    annoy_index_path = os.path.join(datafolder, 'index.ann')
    if os.path.exists(annoy_index_path):
        os.remove(annoy_index_path)
        print(f"Deleted {annoy_index_path}")
    return "OK"

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
        global Current_file
        prev_main()
        Current_file = f"data/{uploaded_file.filename}"
        
        return f"PDF file '{uploaded_file.filename}' uploaded and processed successfully."
    else:
        return "No PDF file received."

@app.route('/toggle_language', methods=['POST'])
def toggle_language():
    language = request.get_json().get('language')
    
    # Perform language toggle operation based on the received language code
    if language == 'en':
        # Code to switch to English language
        pass
    elif language == 'jp':
        # Code to switch to Japanese language
        pass
    
    return 'Language switched successfully'

@app.route('/send')
def getresponse():
    query = request.args.get('query', '')
    language = request.args.get('language', 'en')
    print(f"Query Received: {query} (Language: {language})")
    neighbor = text2embedd2query2(os.path.join(datafolder, 'output.txt'),query)
    combined_string = ' '.join(neighbor)
    input_text = tokenizer("This is my query: " + '\n' + query + ' and this is the context: '+ '\n' + combined_string + '\n' + " Answer the query accordingly ",return_tensors="pt").to(device="cuda")
    print("input text       ", input_text)
    outputs = model.generate(**input_text,**generation_params)
    
    output = tokenizer.decode(outputs[0])
    print(output)
    print(Current_file)
    nearest = text2embedd2query(query, f'{Current_file}')
    nearest_top = nearest[0]
    print("nearest", nearest[0])
    with open(f"{Current_file}_text_xref_pairs.pickle", "rb") as f:
        retrieved = pickle.load(f)

    retrieved_list = []
    index = 0
    image_xref=0
    for i in retrieved:
        image_xref=i[0]
        retrieved_list.append(i[1])
        if nearest[0] == i[1]:
            image_xref = i[0]
            break
        index += 1

    with open(f"{Current_file}_prev_text.pickle", "rb") as f:
        retrieved_list = pickle.load(f)

    with open(f"{Current_file}_text_xref_pairs.pickle", "rb") as f:
        list2 = pickle.load(f)

    xrefs = []
    for i in list2:
        xrefs.append(i[0])

    pair = find_most_similar_sentence(retrieved_list, xrefs, index, image_xref)
    # image_path = display_images(f'{Current_file}', pair)
    image_path = f"images/extracted_image_{Current_file[5:-4]}_{image_xref}.jpg"
    print(image_path)
    # def generate_response():
    #     for line in lines:
    #         yield line + "\n"
    #     if image_path and os.path.exists(image_path):
    #         yield f"{image_path}"

    def generate_response(output: str,image_path):
        output = output[output.find("Answer the query accordingly")+len("Answer the query accordingly"):]
        yield output.encode('utf-8')
        # global image_path
        image_path = r"G:/My Drive/CS671-DL/Hackathon/website/" + image_path
        if image_path and os.path.exists(image_path):
            print("sent")
            with open(image_path, 'rb') as f:
                image_data = f.read()
            base64_image_data = b64encode(image_data).decode('utf-8')
            yield f"data:image/jpeg;base64,{base64_image_data}".encode('utf-8')

    return Response(generate_response(output,image_path), mimetype='text/html')



@app.route('/styles/<path:filename>')
def serve_styles(filename):
    return send_from_directory('styles', filename)

# Serve static files from the 'scripts' directory
@app.route('/scripts/<path:filename>')
def serve_scripts(filename):
    return send_from_directory('scripts', filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500)