import json
nb = json.load(open("Notebook/training/03_Discrete_SAGAN_Production.ipynb", encoding="utf-8"))
print("CELL 8:")
print("".join(nb["cells"][8]["source"]))
print("="*60)
print("CELL 9:")
print("".join(nb["cells"][9]["source"]))

