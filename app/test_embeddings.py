from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="nomic-embed-text")
text = open("data/synthetic/doc01_referral.txt", encoding="utf-8").read()
vector = emb.embed_query(text[:1000])
print(len(vector))
print(vector[:5])