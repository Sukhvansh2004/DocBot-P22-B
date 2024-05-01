
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
 
auth_token = "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"

# sbert_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1', use_auth_token=auth_token)  # Example model, replace with your desired SBERT model

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", use_auth_token=auth_token)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16,
    use_auth_token=auth_token,
)
generation_params = {
    "max_length": 1000
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Move SBERT model to CUDA if available
model.to(device=device)
# Move LM model to CUDA if available
tokenizer.to(device=device)
# Ensure that the LM model is in evaluation mode
model.eval()
input_text = tokenizer(
    "これは私の質問です ヤマハとは何ですか",
    return_tensors="pt")

outputs = model.generate(**input_text, **generation_params)

output = tokenizer.decode(outputs[0])


