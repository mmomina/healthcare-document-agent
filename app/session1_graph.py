from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

llm = ChatOllama(model="qwen3:8b", temperature=0)

class State(TypedDict):
    document_text: str
    doc_type: str

def classify(state: State) -> dict:
    prompt = (
        "Classify this healthcare admin document into ONE of: "
        "referral, discharge_summary, pathology_result, admin_letter, "
        "appointment_request, other. Reply with only the label.\n\n"
        f"Document:\n{state['document_text']}"
    )
    answer = llm.invoke(prompt)
    return {"doc_type": answer.content.strip().lower()}

graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_edge(START, "classify")
graph.add_edge("classify", END)
app_graph = graph.compile()

if __name__ == "__main__":
    sample = open("data/synthetic/doc01_referral.txt", encoding="utf-8").read()
    result = app_graph.invoke({"document_text": sample, "doc_type": ""})
    print(result["doc_type"])   # expect: referral