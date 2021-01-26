from sentence_transformers import SentenceTransformer, util

import torch

model_save_path = "../outputs/hrtps/"
embedder = SentenceTransformer(model_save_path)

# Corpus with example sentences
#Sentences we want sentence embeddings for
corpus = ['谁是公司的投资方',
            '公司几号地铁可以到达',
            '公司的核心价值观',
	    '福利有哪一些',
	    '公司有哪些福利',
	    '公司的办公室在哪里',
            '城南旧事作者简介',
            '在哪里可以在线观看建国大业',
            '年假有几天']

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['公司福利', '公司年假', '公司的投资方', '城南旧事的作者是谁', '网上哪能看建国大业']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(2, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use torch.topk to find the highest 5 scores
    top_results = torch.topk(cos_scores, k=top_k)

    print("======================")
    print("Query:", query)
    print("Top 2 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: %.4f)" % (score))
