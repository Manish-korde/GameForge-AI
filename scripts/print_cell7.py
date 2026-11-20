import json
nb = json.load(open("Notebook/training/03_Discrete_SAGAN_Production.ipynb", encoding="utf-8"))
# Let's find the training loop cell
for idx, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "Starting Production Discrete SAGAN Training" in src or "for epoch in range" in src:
        print(f"Training loop is Cell index: {idx}")
        print(src)

