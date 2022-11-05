from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

inputs = tokenizer("税波是傻逼", return_tensors="pt")
model(**inputs)
