import chromadb
from sentence_transformers import SentenceTransformer


def naive_chunk(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


with open("data/apple_10k.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = naive_chunk(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="apple_10k"
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("Stored chunks:", collection.count())