import PyPDF2
import os
from transformers import AutoModel,AutoTokenizer, AutoModelForCausalLM
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import torch
device='cuda' if torch.cuda.is_available() else 'cpu'
class TextGen:
    def __init__(self):
        self.datafolder = 'data'
        os.makedirs(self.datafolder, exist_ok=True)  # Create the 'data' folder if it doesn't exist

        auth_token = "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"
        self.sbert_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1', use_auth_token=auth_token)
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", use_auth_token=auth_token)
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", torch_dtype=torch.bfloat16, use_auth_token=auth_token)
        self.generation_params = {"max_length": 3000}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sbert_model = self.sbert_model.to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.Jatokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.Jamodel = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
        self.Jamodel.to(device)
        self.Jamodel.eval()

    def pdf2text(self, pdf, id, language, pdf_name):
        annoy_index_path = os.path.join(self.datafolder, id, f'{pdf_name}-{language}.ann')
        text = ""
        with open(pdf, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                
        with open(os.path.join(self.datafolder, id, 'output.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        
        paragraphs=[]
        vector_dimension = None
        if language=="English":
            with open(os.path.join(self.datafolder, id, 'output.txt'), 'r', encoding='utf-8') as file:
                current_paragraph = ""
                for line in file:
                    current_paragraph += line.strip() + " "
                    if '.' in line:
                        if current_paragraph.count('.') >= 5:
                            paragraphs.append(current_paragraph.strip())
                            current_paragraph = ""
            paragraph_embeddings = [np.array(self.sbert_model.encode([paragraph])[0]) for paragraph in paragraphs]
            vector_dimension = len(paragraph_embeddings[0])
            annoy_index = AnnoyIndex(vector_dimension, 'angular')
            for i, vector in enumerate(paragraph_embeddings):
                annoy_index.add_item(i, vector)
            annoy_index.build(n_trees=15)
            self.save_annoy_index_to_file(annoy_index, annoy_index_path)
            
        elif language=="Japanese":
            with open(os.path.join(self.datafolder, id, 'output.txt'), 'r', encoding='utf-8') as file:
                current_paragraph = ""
                for line in file:
                    current_paragraph += line.strip() + " "
                    if '.' in line:
                        if current_paragraph.count('.') >= 3:
                            paragraphs.append(current_paragraph.strip())
                            current_paragraph = ""
            paragraph_embeddings=[]
            for paragraph in paragraphs:
                paragraph_embedding = self.vectorize_paragraph(paragraph)
                if len(paragraph_embedding) > 768:
                    paragraph_embedding = paragraph_embedding[:768]
                paragraph_embeddings.append(paragraph_embedding)
            vector_dimension=paragraph_embeddings[0].shape[0]

            annoy_index=AnnoyIndex(vector_dimension,'angular')
            for i,vector in enumerate(paragraph_embeddings):
                annoy_index.add_item(i,vector)
            annoy_index.build(n_trees=15)
            self.save_annoy_index_to_file(annoy_index, annoy_index_path)
            
        print(f"PDF file '{pdf_name}' saved to '{pdf}'")
            
        return paragraphs, vector_dimension
                
    def vectorize_paragraph(self, paragraph, max_length=512):
        tokenized_paragraph = self.Jatokenizer([paragraph], return_tensors="pt").to(device)
        tokenized_chunks = []
        for i in range(0, tokenized_paragraph['input_ids'].size(1), max_length):
            chunk = {key: value[:, i:i+max_length] for key, value in tokenized_paragraph.items()}
            tokenized_chunks.append(chunk)

        chunk_embeddings = []
        for chunk in tokenized_chunks:
            with torch.no_grad():
                outputs = self.Jamodel(**chunk)
                chunk_embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())
        paragraph_embedding = np.concatenate(chunk_embeddings, axis=0)
        return paragraph_embedding
    
    def text2embedd2query(self, query, language, paragraphs, id, pdf_name, vector_dim):

        num_neighbors = 5
        if language=="English":
            query_embedding = self.sbert_model.encode([query])[0]
        elif language=="Japanese":
            query_embedding = self.vectorize_paragraph(query)
        
        annoy_index_path = os.path.join(self.datafolder, id, f'{pdf_name}-{language}.ann')
        annoy_index=self.load_annoy_index_from_file(vector_dim, annoy_index_path)
        nearest_neighbor_indices = annoy_index.get_nns_by_vector(query_embedding, num_neighbors)
        nearest_neighbor_paragraphs = [paragraphs[index] for index in nearest_neighbor_indices]
        return nearest_neighbor_paragraphs

    def save_annoy_index_to_file(self, annoy_index, file_path):
        try:
            annoy_index.save(file_path)
        except:
            pass

    def load_annoy_index_from_file(self, vector_dimension, file_path):
        if os.path.exists(file_path):
            annoy_index = AnnoyIndex(vector_dimension, 'angular')
            annoy_index.load(file_path)
            return annoy_index
        else:
            return None

    def generate_response(self, output):
        output = output[output.find("Answer the query accordingly") + len("Answer the query accordingly"):]
        return output.encode('utf-8')

    def process_pdf(self, uploaded_pdf, language, id):
        file_path = os.path.join(self.datafolder, id, uploaded_pdf.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_pdf.read())
            
        return self.pdf2text(file_path, id, language, uploaded_pdf.name)
        
    def process_query(self, query, language, paragraphs, id, pdf_name, vector_dim):
        # Process the query using text2embedd2query and generate_response
        neighbor = self.text2embedd2query(query, language, paragraphs, id, pdf_name, vector_dim)
        combined_string = ' '.join(neighbor)
        if language == "English":
            input_text = self.tokenizer(
                "This is your query: " + '\n' + query + 'This is your Data:' + '\n' + combined_string + '\n' + " Answer the query accordingly ",
                return_tensors="pt").to(device=self.device)
        elif language == "Japanese":
            input_text = self.tokenizer(
                "これがあなたのクエリです:"+'\n'+ query + "これはあなたのデータです:" +'\n' + combined_string + '\n' + "それに応じて質問に答えます ",
                return_tensors="pt").to(device=self.device)
        outputs = self.model.generate(**input_text, **self.generation_params)
        output = self.tokenizer.decode(outputs[0])
        return self.generate_response(output)
