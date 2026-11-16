from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langsmith import traceable
import sys

from agentic_rag import app_graph as qa_graph
from session2_classifier import app_graph as classifier_graph

from dotenv import load_dotenv

load_dotenv()

class WorkflowState(TypedDict):
    input_text: str    # the document or question that arrived
    route: str         # "classify" or "answer", set by the route node
    route_reason: str  # why the router chose that route, printed on every run
    result: str        # the final output text, whichever agent produced it
    
class Route(BaseModel):
    route: Literal["classify", "answer"] = Field(
        description="classify if the input is a document to file, answer if it is a question about the documents"
    )
    reason: str = Field(description="one short sentence saying why")
    
llm = ChatOllama(model="qwen3:8b", temperature=0)
router = llm.with_structured_output(Route)

@traceable(name="extract_input")
def extract(state: WorkflowState) -> dict:
    return {"input_text": state["input_text"]}  # just pass it through, but trace it

@traceable(name="route_input")
def route(state: WorkflowState) -> dict:
    prompt = (
    "You are routing input for a clinic document system. "
    "Decide whether the input is a document to be filed, or a question about the documents.\n\n"
    f"Input:\n{state['input_text']}"
    )
    decision = router.invoke(prompt)
    return { "route": decision.route, "route_reason": decision.reason }

@traceable(name="run_classifier")
def run_classifier(state: WorkflowState) -> dict:
    result = classifier_graph.invoke({
    "document_text": state["input_text"],
    "classification": {},
    "flags": [],
    })

    return {"result": str(result["classification"])}

@traceable(name="run_qa")
def run_qa(state: WorkflowState) -> dict:
    result = qa_graph.invoke({
        "question": state["input_text"],
        "original_question": state["input_text"],
        "chunks": [],
        "answer": "",
        "loops": 0,
        "feedback": "",
        "check_loops": 0,
    })
    return {"result": result["answer"]}


builder = StateGraph(WorkflowState)
builder.add_node("extract", extract)
builder.add_node("route", route)
builder.add_node("classify", run_classifier)
builder.add_node("answer", run_qa)

builder.add_edge(START, "extract")
builder.add_edge("extract", "route")
builder.add_conditional_edges("route", lambda state: state["route"], {"classify": "classify", "answer": "answer"})
builder.add_edge("classify", END)
builder.add_edge("answer", END)

workflow = builder.compile()

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Why does Priya Nair need urgent review?"
    result = workflow.invoke({"input_text": q})
    print("ROUTE:", result["route"], "-", result["route_reason"])
    print("\nRESULT:\n", result["result"])
    