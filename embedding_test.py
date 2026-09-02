from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")
text="Apple revenue increased significantly in fiscal year 2025"
embedding=model.encode(text)
print("Embedding type:", type(embedding))
print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])