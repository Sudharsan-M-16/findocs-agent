def naive_chunk(text,chunk_size=1000,overlap=200):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        if chunk.strip():
            chunks.append(chunk)
            
        start+=chunk_size-overlap
    return chunks
with open("data/apple_10k.txt","r",encoding="utf-8") as f:
    text=f.read()
chunks=naive_chunk(text)
print("Total characters:",len(text))
print("Number of chunks:",len(chunks))
print("First chunk:")
print(chunks[0])

            