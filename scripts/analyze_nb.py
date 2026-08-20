import json

with open("Notebook/280K_Autoencoder_Evaluation.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(12, 20):
    cell = nb['cells'][i]
    source_lines = cell.get("source", [])
    print(f"cell_{i}_source = " + '"""\\\n' + "".join(source_lines).replace('\\', '\\\\').replace('"', '\\"') + '"""\n')

