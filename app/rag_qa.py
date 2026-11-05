import sys
from typing import TypedDict, List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END
from langsmith import traceable


embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = store.as_retriever(search_kwargs={"k": 4})
llm = ChatOllama(model="qwen3:8b", temperature=0)

#@traceable(name="rag_qa")

class State(TypedDict):
    question: str
    chunks: List[Document]
    answer: str


def retrieve(state: State) -> dict:
    return {"chunks": retriever.invoke(state["question"])}

@traceable(name="rag_qa")
def answer(state: State) -> dict:
    context = "\n\n".join(
        f"[Source: {c.metadata['source']}]\n{c.page_content}" for c in state["chunks"]
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Name the source document your answer comes from. "
        "If the context does not contain the answer, say: "
        "I cannot find that in these documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}"
    )
    return {"answer": llm.invoke(prompt).content}


graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)
app_graph = graph.compile()

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Why does Priya Nair need urgent review?"
    result = app_graph.invoke({"question": q, "chunks": [], "answer": ""})
    print(result["answer"])
    print("\nSources retrieved:")
    for c in result["chunks"]:
        print(" -", c.metadata["source"])