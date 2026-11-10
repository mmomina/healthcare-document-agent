from agentic_rag import app_graph

graph = app_graph.get_graph()

print(graph.draw_mermaid())

with open("graph.png", "wb") as f:
    f.write(graph.draw_mermaid_png())

print("saved graph.png")
