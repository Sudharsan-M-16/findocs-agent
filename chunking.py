import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def detect_heading(line):
    line = line.strip()

    if re.fullmatch(r"PART\s+[IVX]+", line, re.IGNORECASE):
        return "PART"

    if re.fullmatch(r"Item\s+\d+[A-Z]?\..*", line, re.IGNORECASE):
        return "ITEM"

    return None


def parse_sections(text):
    lines = text.splitlines()

    sections = []
    current_heading = "Document Start"
    current_lines = []

    for line in lines:
        heading_type = detect_heading(line)

        if heading_type:
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines)
                })

            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_lines)
        })

    return sections


def chunk_sections(sections):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    chunks = []

    for section in sections:
        section_chunks = splitter.split_text(section["text"])

        for chunk in section_chunks:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "company": "Apple Inc.",
                    "filing_date": "2025-10-31",
                    "section": section["heading"]
                }
            })

    return chunks


with open("data/apple_10k.txt", "r", encoding="utf-8") as f:
    text = f.read()

sections = parse_sections(text)
chunks = chunk_sections(sections)

print("Number of sections:", len(sections))
print("Number of chunks:", len(chunks))

print("\nFIRST 10 SECTIONS")
print("=" * 80)

for section in sections[:10]:
    print(section["heading"])

print("\nFIRST CHUNK")
print("=" * 80)
print(chunks[0]["text"])

print("\nFIRST CHUNK METADATA")
print("=" * 80)
print(chunks[0]["metadata"])