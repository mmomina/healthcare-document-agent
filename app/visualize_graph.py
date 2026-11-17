from clinic_workflow import workflow

graph = workflow.get_graph()

print(graph.draw_mermaid())

with open("graph.png", "wb") as f:
    f.write(graph.draw_mermaid_png())

print("saved graph.png")
