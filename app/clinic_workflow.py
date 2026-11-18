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
    input_text: str      # what the user typed this turn
    resolved_text: str   # standalone version, set by the resolve node
    route: str           # "classify" or "answer", set by the route node
    route_reason: str    # why, set by the route node
    result: str          # what comes back from whichever agent ran
    history: list        # past turns: [{"input": ..., "route": ..., "result": ...}]

class Route(BaseModel):
    reason: str = Field(description="one short sentence saying why")
    route: Literal["classify", "answer"] = Field(
        description="classify if the input is a document to file, answer if it is a question about the documents"
    )
llm = ChatOllama(model="qwen3:8b", temperature=0)
router = llm.with_structured_output(Route)

class Resolved(BaseModel):
    resolved: str = Field(description="the input rewritten as a standalone text, or unchanged")
    was_followup: bool = Field(description="true if the input referred to an earlier turn")
resolver = llm.with_structured_output(Resolved)


@traceable(name="extract_input")
def extract(state: WorkflowState) -> dict:
    return {"input_text": state["input_text"]}  # just pass it through, but trace it

@traceable(name="resolve_input")
def resolve(state: WorkflowState) -> dict:
    if not state["history"]:
        return {"resolved_text": state["input_text"]}
    history_text = "\n".join(
        f"Turn {i + 1} input: {h['input']}\nTurn {i + 1} result: {h['result']}"
        for i, h in enumerate(state["history"])
    )
    prompt = (
        "You prepare input for a clinic document system. "
        "Rewrite the new input as a standalone text using the conversation history, "
        "only if it refers to something earlier (words like her, him, that, it, the same patient). "
        "Replace the reference with the actual name or detail from the history. "
        "If the new input already stands alone, return it unchanged. "
        "Never add facts that are not in the history or the new input.\n\n"
        f"History:\n{history_text}\n\nNew input:\n{state['input_text']}"
    )
    out = resolver.invoke(prompt)
    if out.was_followup:
        print(f"RESOLVED: {out.resolved}")
    return {"resolved_text": out.resolved}


@traceable(name="route_input")
def route(state: WorkflowState) -> dict:
    prompt = (
        "You are routing input for a clinic document system. "
        "Ask one question first: is the writer handing over a record, or asking for information back? "
        "If they are handing over a record, a letter, referral, report or discharge summary written about a patient, route classify, even when it contains requests like 'please arrange'. "
        "If they expect an answer back, route answer. "
        "Examples: 'Pathology result: HbA1c 7.8%. Comment: above target range, GP review recommended.' -> classify. "
        "'Appointment confirmation: your appointment with Dr Lee is booked for Tuesday 10:30am. Please arrive 10 minutes early.' -> classify. "
        "'What follow-up was arranged for Mr Chen after his procedure?' -> answer. "
        "'What medication was prescribed at the last visit?' -> answer. "
        f"Input:\n{state['resolved_text']}"
    )
    decision = router.invoke(prompt)
    return {"route": decision.route, "route_reason": decision.reason}

@traceable(name="run_classifier")
def run_classifier(state: WorkflowState) -> dict:
    result = classifier_graph.invoke({
    "document_text": state["resolved_text"],
    "classification": {},
    "flags": [],
    })

    return {"result": str(result["classification"])}

@traceable(name="run_qa")
def run_qa(state: WorkflowState) -> dict:
    result = qa_graph.invoke({
        "question": state["resolved_text"],
        "original_question": state["resolved_text"],
        "chunks": [],
        "answer": "",
        "loops": 0,
        "feedback": "",
        "check_loops": 0,
    })
    return {"result": result["answer"]}

@traceable(name="record_turn")
def record(state: WorkflowState) -> dict:
    turn = {
        "input": state["input_text"],
        "route": state["route"],
        "result": state["result"],
    }
    return {"history": (state["history"] + [turn])[-5:]}

builder = StateGraph(WorkflowState)
builder.add_node("extract", extract)
builder.add_node("resolve", resolve)
builder.add_node("route", route)
builder.add_node("classify", run_classifier)
builder.add_node("answer", run_qa)
builder.add_node("record", record)

builder.add_edge(START, "extract")
builder.add_edge("extract", "resolve")
builder.add_edge("resolve", "route")
builder.add_conditional_edges("route", lambda state: state["route"], {"classify": "classify", "answer": "answer"})
builder.add_edge("classify", "record")
builder.add_edge("answer", "record")
builder.add_edge("record", END)
workflow = builder.compile()

if __name__ == "__main__":
    history = []
    print("Clinic workflow. Type a document or a question. Type exit to stop.")
    while True:
        q = input(">> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        result = workflow.invoke({
            "input_text": q,
            "resolved_text": "",
            "route": "",
            "route_reason": "",
            "result": "",
            "history": history,
        })
        history = result["history"]
        print("RESULT:")
        print(" ", result["result"])
    