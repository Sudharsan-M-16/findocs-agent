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

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("First chunk length:", len(chunks[0]))
print("First embedding:", embeddings[0][:10])