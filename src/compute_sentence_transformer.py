
from sentence_transformers import SentenceTransformer
import numpy as np


model_save_path = "../outputs/taobao/"
# Load pre-trained Sentence Transformer Model (based on DistilBERT). It will be downloaded automatically
model = SentenceTransformer(model_save_path, device='cuda')

# Embed a list of sentences
#Sentences we want sentence embeddings for
sentences = ['谁是公司的投资方',
            '公司几号地铁可以到达',
            '公司的核心价值观']

sentence_embeddings = model.encode(sentences)

# The result is a list of sentence embeddings as numpy arrays
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
