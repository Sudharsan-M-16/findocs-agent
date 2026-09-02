import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection("apple_10k")

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "Why did Apple's Services net sales increase in 2025?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

for i, document in enumerate(results["documents"][0]):
    print("\n" + "=" * 80)
    print(f"RESULT {i + 1}")
    print("=" * 80)
    print(document)