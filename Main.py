import streamlit as st
from transformers import AutoTokenizer
import transformers
import torch
from transformers import pipeline
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
os.environ["HF_TOKEN"]="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"
model = "microsoft/phi-2" # meta-llama/Llama-2-7b-hf
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
llama_pipeline = pipeline(
  "text-generation",  # LLM task
  model=model,
  torch_dtype=torch.float32,
  device_map="auto",
)
llama_pipeline(res)
  
st.title("Phi-2")
col1,col2 = st.columns(2)
ques = col1.text_input("Enter the Query : ")
key = col2.text_input("Enter the Key : ")
res = st.button("Submit")
if res :
    str = llama_pipeline(res)
    st.write(str)
