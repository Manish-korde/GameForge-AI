import json
import ast

nb_path = "Notebook/training/03_Discrete_SAGAN_Production.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        try:
            ast.parse(code)
            print(f"Cell {idx}: VALID")
        except SyntaxError as e:
            print(f"Cell {idx}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
            print(f"Code snippet around error:")
            lines = code.split("\n")
            for li in range(max(0, e.lineno - 3), min(len(lines), e.lineno + 3)):
                print(f"  {li+1}: {lines[li]}")
            raise

