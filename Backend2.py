import pikepdf
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer , util
import pickle
import fitz
import os
from PIL import Image
import io

auth_token = "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"
sbert_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1', use_auth_token=auth_token) 
datafolder = 'data'
os.makedirs(datafolder, exist_ok=True)

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
    
    for xref, image_data in images.items():
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
    
def text2embedd2query(query, pdf_path, language):
    with open(f"{pdf_path}_prev_text.pickle", "rb") as f:
        retrieved_list = pickle.load(f)
    sentences = retrieved_list
    sentences.reverse()
    paragraphs = [sentence for sentence in sentences]
    paragraph_embeddings = [np.array(sbert_model.encode([paragraph])[0]) for paragraph in paragraphs]

    vector_dimension = len(paragraph_embeddings[0])
    
    annoy_index_path = os.path.join(datafolder, f'index_{language}.ann')
    annoy_index = None # load_annoy_index_from_file(vector_dimension, annoy_index_path)
    if annoy_index is None:
        annoy_index = AnnoyIndex(vector_dimension, 'angular')
        for i, vector in enumerate(paragraph_embeddings):
            annoy_index.add_item(i, vector)
        annoy_index.build(n_trees=15)
        # save_annoy_index_to_file(annoy_index, annoy_index_path)
        
    num_neighbors = 5
    query_embedding = sbert_model.encode([query])[0]
    num_neighbors = 5

    nearest_neighbor_indices = annoy_index.get_nns_by_vector(query_embedding, num_neighbors)

    nearest_neighbor_paragraphs = [paragraphs[index] for index in nearest_neighbor_indices]
    
    # nearest_neighbor_paragraphs = " Hello "
    return nearest_neighbor_paragraphs

def save_annoy_index_to_file(annoy_index: AnnoyIndex, file_path):
    try:
        annoy_index.save(file_path)
    except:
        pass

def load_annoy_index_from_file(vector_dimension, file_path):
    if os.path.exists(file_path):
        annoy_index = AnnoyIndex(vector_dimension, 'angular')
        annoy_index.load(file_path)
        return annoy_index
    else:
        return None
def unlock_pdf(pdf_path):
    filename = pdf_path

    with pikepdf.open(filename, allow_overwriting_input=True) as pdf:
        print(f"File {filename} opened correctly.")
        pdf.save(filename)
    print(f"File {filename} unlocked.")
    
def process_pdf(pdf_path):
    # pdf_files = list_pdf_files_glob(datafolder)

    # for pdf_path in pdf_files:
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
        
def save_data(pdf_path, text_xref_pairs_list_1, text_list_1):
    with open(f"{pdf_path}_text_xref_pairs.pickle", "wb") as f:
        pickle.dump(text_xref_pairs_list_1, f)

    with open(f"{pdf_path}_prev_text.pickle", "wb") as f:
        pickle.dump(text_list_1, f)
        
def main_current(query, pdf_path, language):
    # pdf_files = list_pdf_files_glob(datafolder)

    # for pdf_path in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        # query = "Instrument and  control functions  4 - 3 3s"
        process_pdf(pdf_path)
        nearest = text2embedd2query(query, pdf_path, language)
        print(nearest)
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

        return display_images(pdf_name, pair)
    
def display_images(pdf_name, pair):
    img = []
    for i in pair:
        img.append(os.path.join("images",f'extracted_image_{pdf_name}_{i}.jpg'))
    return img