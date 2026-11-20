import json

with open("alucard_frontfacing_42k_indices.json", "r") as f:
    indices = json.load(f)

print(f"Total front-facing indices in file: {len(indices)}")

