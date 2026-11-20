import json

nb_path = "Notebook/training/03_Discrete_SAGAN_Production.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell_idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            # Replace \"\"\" or \" with regular quotes
            fixed_line = line.replace('\\"', '"')
            new_source.append(fixed_line)
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Scanned and cleaned ALL cells in 03_Discrete_SAGAN_Production.ipynb!")

