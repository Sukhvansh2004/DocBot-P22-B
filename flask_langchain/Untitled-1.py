import pdfbox
from flask import Response
import os
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import jsonify
import transformers
import torch 



auth_token = os.environ.get("HF_TOKEN")  # was a hard-coded token; export HF_TOKEN instead

sbert_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1',use_auth_token=auth_token)  # Example model, replace with your desired SBERT model

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

print(pipeline("hey i am stupid"))