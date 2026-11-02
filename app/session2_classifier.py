import re
from pathlib import Path
from typing import TypedDict, List, Literal

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


class Classification(BaseModel):
    doc_type: Literal["referral", "discharge_summary", "pathology_result",
                      "admin_letter", "appointment_request", "other"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    urgency: Literal["routine", "soon", "urgent"]


llm = ChatOllama(model="qwen3:8b", temperature=0)
structured_llm = llm.with_structured_output(Classification)


class State(TypedDict):
    document_text: str
    classification: dict
    flags: List[str]


def classify(state: State) -> dict:
    prompt = (
        "Classify this healthcare admin document and fill the form. "
        "urgency: urgent means needs action within days or has red flag symptoms, "
        "soon means review within 2 to 3 weeks, routine means no time pressure.\n\n"
        f"Document:\n{state['document_text']}"
    )
    result = structured_llm.invoke(prompt)
    return {"classification": result.model_dump()}


def flag_pii(state: State) -> dict:
    text = state["document_text"]
    flags = []
    if re.search(r"DOB", text):
        flags.append("date_of_birth")
    if re.search(r"Medicare", text):
        flags.append("medicare_number")
    if re.search(r"\b04\d{8}\b", text.replace(" ", "")):
        flags.append("phone_number")
    for word in ["urgent", "abnormal", "follow-up"]:
        if word in text.lower():
            flags.append(f"sensitive_word:{word}")
    return {"flags": sorted(set(flags))}


graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("flag_pii", flag_pii)
graph.add_edge(START, "classify")
graph.add_edge("classify", "flag_pii")
graph.add_edge("flag_pii", END)
app_graph = graph.compile()


if __name__ == "__main__":
    #files = sorted(Path("data/synthetic").glob("doc0[1-6]*.txt")) #session2
    files = sorted(Path("data/synthetic").glob("doc*.txt")) #session3
    print(files)
    for f in files:
        state = {"document_text": f.read_text(encoding="utf-8"),
                 "classification": {}, "flags": []}
        result = app_graph.invoke(state)
        c = result["classification"]
        print(f"{f.name} | {c['doc_type']} | conf {c['confidence']} | "
              f"{c['urgency']} | flags: {result['flags']} | {c['reason']}")