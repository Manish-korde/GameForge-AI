import json
import ast

nb_path = "Notebook/training/02_DCGAN_Discrete_Continuation_Epochs51_to_100.ipynb"
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
            raise

