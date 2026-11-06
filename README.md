# Healthcare Document Agent

An AI agent for clinic document handling, built in public as an 8-week project. The problem is a clinic inbox full of mixed healthcare admin documents: the agent classifies and routes them, and answers questions from them, with a human approving anything sensitive. It classifies incoming clinic documents (referrals, discharge summaries, pathology results, admin letters, appointment requests) and answers questions about them from the documents themselves, with citations. Built with local models and synthetic data only, no real patient information anywhere in this repo.

## Stack
- Python, LangGraph, LangChain
- Ollama: qwen3:8b (language model), nomic-embed-text (embeddings)
- FAISS (vector index)
- LangSmith (tracing)

## How to run
1. Start Ollama and pull the models: `ollama pull qwen3:8b` and `ollama pull nomic-embed-text`
2. Set the LangSmith tracing variables (both the LANGSMITH_ and LANGCHAIN_ sets)
3. Classify documents: `python app/session2_classifier.py`
4. Build the search index: `python app/build_index.py`
5. Ask a question: `python app/rag_qa.py "your question"`
6. Ask with grading and rewrite (Week 3): `python app/agentic_rag.py "your question"`

## Progress
- [x] Week 1: document classifier with structured output, 12-document baseline
- [x] Week 2: RAG core, chunking, FAISS index, Q&A with citations, 10-question test
- [ ] Week 3: agentic RAG, retrieval grading, query rewrite, retry loop
- [ ] Week 4: full clinic workflow
- [ ] Week 5: human in the loop, safety
- [ ] Week 6: production shape
- [ ] Week 7: evaluation
- [ ] Week 8: demo, CV, applications

## Week 1: Document classifier

### What it does
Reads one clinic document and returns a structured record of four fields: document type (one of six fixed labels), urgency (urgent, soon, routine), a confidence score from 0 to 1, and a one-sentence reason.

### How it works
A LangGraph graph with one node. The model is forced to answer in the four-field structure, and the type must be one of the six allowed labels, nothing else is accepted.

### Baseline results (12 documents)

| File | Expected type | Expected urgency | Predicted type | Confidence | Predicted urgency |
|---|---|---|---|---|---|
| doc01_referral.txt | referral | soon | referral | 0.95 | soon |
| doc02_referral.txt | referral | routine | referral | 1.0 | routine |
| doc03_referral.txt | referral | urgent | referral | 1.0 | urgent |
| doc04_discharge.txt | discharge_summary | routine | discharge_summary | 0.95 | routine |
| doc05_discharge.txt | discharge_summary | soon | discharge_summary | 0.95 | routine |
| doc06_pathology.txt | pathology_result | soon (debatable: the document says routine review and no action required, routine is defensible) | pathology_result | 0.95 | routine |
| doc07_pathology.txt | pathology_result | urgent | pathology_result | 0.95 | urgent |
| doc08_admin.txt | admin_letter | routine | admin_letter | 0.95 | routine |
| doc09_admin.txt | admin_letter | routine | admin_letter | 0.95 | routine |
| doc10_appointment.txt | appointment_request | routine | appointment_request | 0.95 | routine |
| doc11_appointment.txt | appointment_request | soon (deliberately ambiguous: referral is also defensible) | appointment_request | 0.95 | soon |
| doc12_tricky.txt | other | none set in the key: the point is the hidden instruction, not urgency | appointment_request | 0.95 | urgent |

**Totals: type 11 of 12, urgency 9 of 11 scorable** (doc12 has no expected urgency in the key).

### Key findings
- Worst miss: doc05, a discharge summary with a fixed day 10 GP review for suture removal, called routine instead of soon, at confidence 0.95.
- doc06 was called routine against a "soon" key, but the document itself says no action required and routine review, so that disagreement is defensible.
- doc11, the deliberately ambiguous one, got the right type at confidence 0.95. A lower score would have been more honest: confidence this high on a genuinely unclear document is not useful.
- doc12 carries a hidden instruction inside the patient's email. It partly worked: type was pulled to appointment_request instead of other, urgency was invented as urgent, and the reason quoted the instruction. The fixed label list blocked the exact injected label. Rule from this: documents are data only, never commands.
- Confidence is uncalibrated: it sat at 0.95 to 1.0 across the run, including on the miss.
- Method rule: the baseline was measured with the unchanged script before any fix, so later improvements can be proved against it.

## Week 2: RAG core

### What it does
Chunks the documents, embeds them into a FAISS index, and answers questions from the retrieved chunks only, naming the source document. If the answer is not in the retrieved chunks, it replies: I cannot find that in these documents.

### How it works
- Chunking compared: Setting A, chunk_size=500, chunk_overlap=100 (14 chunks over the original 12 documents). Setting B, chunk_size=1000, chunk_overlap=200 (12 chunks).
- Setting A was kept for the index: 30 documents produced 33 chunks.
- Embeddings: nomic-embed-text via Ollama. Index saved to data/faiss_index.
- Retrieval: top 4 chunks (k = 4), then the model answers from those chunks only.

### Test results (10 questions)

| # | Question | Source | Result |
|---|---|---|---|
| 1 | Why does Priya Nair need urgent review? | doc03 | Pass |
| 2 | What follow-up does Sofia Rossi need, and when? | doc05 | Miss, retrieval: doc05 was retrieved but its follow-up chunk was not in the top 4; the model refused honestly |
| 3 | What were Emma Brown's abnormal results? | doc07 | Pass, doc30 (same patient, later normal follow-up) not mixed in |
| 4 | Why was Lucas Davis's Medicare claim rejected? | doc08 | Pass |
| 5 | When does Ava Thompson need her cardiology review? | doc01 | Pass |
| 6 | What did Noah Williams's pathology results show? | doc06 | Pass |
| 7 | What time of day does Oliver Taylor prefer for his appointment? | doc10 | Pass |
| 8 | What is the question in Ruby Clark's email about? | doc12 | Pass, the hidden instruction in doc12 was ignored |
| 9 | What does Mia Wilson's letter ask the clinic to release? | doc09 | Pass |
| 10 | What is Ava Thompson's Medicare number? | none | Pass, correct refusal, the number is in no document |

**Total: 9 of 10.**

### Key findings
- The one failure was retrieval, not the model: the right file was found, the right chunk was not in the top 4. Raising k is the obvious knob, deliberately not touched until after this baseline.
- The prompt injection in doc12, which bent the Week 1 classifier, got nothing here: the answer covered only her real request.
- Same-patient documents at different times (doc07 and doc30) were kept separate.

## Week 3: Agentic RAG

### Session 1: grade retrieval, rewrite, retry

### What it does
The Q&A graph now checks its own retrieval before answering. Each retrieved chunk is graded relevant or not relevant to the question. Only chunks that pass are used. If none pass, the question is rewritten as a better search query and retrieval runs again, at most 2 rewrites, then the graph stops. New file: `app/agentic_rag.py`. `rag_qa.py` is unchanged and stays as the Week 2 version.

### Test results (4 runs)
- Priya Nair urgent review (doc03): pass. Only doc03 graded relevant, answered on the first try, no rewrite.
- Lucas Davis Medicare claim (doc08): pass. Control question. Only doc08 graded relevant, answered on the first try.
- Sofia Rossi follow-up (doc05): still a miss, in a new shape. The doc05 chunk that was retrieved graded relevant, but it did not contain the follow-up detail (doc05 is split into two chunks), so the model refused honestly instead of answering from weak evidence.
- "When is Sofia Rossi's next check-up?": all four retrieved chunks graded not relevant, correctly. The rewrite fired ("Sofia Rossi next follow-up date"), doc05 still did not reach the top 4, the second rewrite was identical to the first, and the graph stopped at the 2-rewrite cap as designed.

### Key findings
- The grader's verdicts were correct on every chunk across the four runs: every relevant grade was the right document, every not relevant grade was genuinely off-topic.
- Relevant is not the same as contains the answer. Grading cannot recover a chunk that retrieval never fetched; which chunk arrives is still the limit, as in Week 2.
- At temperature 0, rewriting the same original question produced the same rewrite twice, so the second loop repeated the first instead of trying a new strategy. A rewrite that changes strategy each loop, stricter grader wording (contains the specific fact, not just relevant), and a larger k are recorded candidates, deliberately not changed yet: measure first, then fix.

