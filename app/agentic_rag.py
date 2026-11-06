import sys
from typing import TypedDict, List, Literal
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END
from langsmith import traceable


embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = store.as_retriever(search_kwargs={"k": 4})
llm = ChatOllama(model="qwen3:8b", temperature=0)

class State(TypedDict):
    question: str            # current search wording, the rewrite node may change this
    original_question: str   # what the person actually asked, never changed
    chunks: List[Document]
    answer: str
    loops: int
    
@traceable(name="agentic_retrieve")
def retrieve(state: State) -> dict:
    chunks = retriever.invoke(state["question"])
    print(f"\nRetrieve (loop {state['loops']}): searched '{state['question']}'")
    for c in chunks:
        print("  found:", c.metadata["source"])
    return {"chunks": chunks}

class Grade(BaseModel):
    binary_score: Literal["yes", "no"]


grader = llm.with_structured_output(Grade)


@traceable(name="agentic_grade")
def grade(state: State) -> dict:
    kept = []
    print("Grades:")
    for c in state["chunks"]:
        prompt = (
            "You are grading whether a document chunk is relevant to a question.\n"
            f"Question: {state['original_question']}\n\n"
            f"Chunk:\n{c.page_content}\n\n"
            "Answer yes if the chunk contains information that helps answer the question, "
            "no if it does not."
        )
        result = grader.invoke(prompt)
        print(f"  {c.metadata['source']}: {result.binary_score}")
        if result.binary_score == "yes":
            kept.append(c)
    return {"chunks": kept}

@traceable(name="agentic_rewrite")
def rewrite(state: State) -> dict:
    prompt = (
        "Rewrite this question as a better search query for finding the answer "
        "in clinic documents such as referral letters, discharge summaries and "
        "pathology reports. Keep the patient name. Use the wording such documents "
        "themselves would use. Output only the rewritten query.\n\n"
        f"Question: {state['original_question']}"
    )
    new_q = llm.invoke(prompt).content.strip()
    print(f"Rewrite {state['loops'] + 1}: '{new_q}'")
    return {"question": new_q, "loops": state["loops"] + 1}

@traceable(name="agentic_answer")
def answer(state: State) -> dict:
    context = "\n\n".join(
        f"[Source: {c.metadata['source']}]\n{c.page_content}" for c in state["chunks"]
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Name the source document your answer comes from. "
        "If the context does not contain the answer, say: "
        "I cannot find that in these documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['original_question']}"
    )
    return {"answer": llm.invoke(prompt).content}


def give_up(state: State) -> dict:
    return {"answer": "No relevant chunk found after 2 rewrites. (Proper refusal path comes in Session 2.)"}


def decide_after_grade(state: State) -> str:
    if len(state["chunks"]) > 0:
        return "answer"
    if state["loops"] < 2:
        return "rewrite"
    return "give_up"


graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("rewrite", rewrite)
graph.add_node("answer", answer)
graph.add_node("give_up", give_up)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", decide_after_grade, {
    "answer": "answer",
    "rewrite": "rewrite",
    "give_up": "give_up",
})
graph.add_edge("rewrite", "retrieve")
graph.add_edge("answer", END)
graph.add_edge("give_up", END)
app_graph = graph.compile()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Why does Priya Nair need urgent review?"
    result = app_graph.invoke({
        "question": q,
        "original_question": q,
        "chunks": [],
        "answer": "",
        "loops": 0,
    })
    print("\nANSWER:\n", result["answer"])
    print("\nSources used:")
    for c in result["chunks"]:
        print(" -", c.metadata["source"])