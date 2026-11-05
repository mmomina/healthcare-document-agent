from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) #tried 1000/200, nothing splits bcs synthetic documents are short, most are under 500 characters of body text, so there is almost nothing to cut 

all_chunks = []
for f in sorted(Path("data/synthetic").glob("doc*.txt")):
    docs = TextLoader(str(f), encoding="utf-8").load()
    chunks = splitter.split_documents(docs)
    print(f"{f.name}: {len(chunks)} chunks")
    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")
print("First chunk of doc03:")
for c in all_chunks:
    if "doc03" in c.metadata["source"]:
        print(c.page_content)
        break