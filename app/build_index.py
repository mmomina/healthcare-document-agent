from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

@traceable(name="build_index")
def main():


    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    all_chunks = []
    for f in sorted(Path("data/synthetic").glob("doc*.txt")):
        docs = TextLoader(str(f), encoding="utf-8").load()
        all_chunks.extend(splitter.split_documents(docs))

    print(f"Embedding {len(all_chunks)} chunks...")
    store = FAISS.from_documents(all_chunks, embeddings)
    store.save_local("data/faiss_index")
    print("Index saved to data/faiss_index")

    results = store.similarity_search("Which patient needs urgent neurology review?", k=3)
    for r in results:
        print(r.metadata["source"], "|", r.page_content[:80].replace("\n", " "))
    
    return {"chunks": len(all_chunks), "index": "data/faiss_index"}

if __name__ == "__main__":
    main()
