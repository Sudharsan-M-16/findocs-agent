from bs4 import BeautifulSoup

with open("data/apple_10k.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

for tag in soup(["script", "style"]):
    tag.decompose()

text = soup.get_text(separator="\n")

with open("data/apple_10k.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("HTML size:", len(html))
print("Text size:", len(text))
