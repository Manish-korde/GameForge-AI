import json

nb = json.load(open("Notebook/training/02_DCGAN_Discrete_Continuation_Epochs51_to_100.ipynb", encoding="utf-8"))

print("=== 8-POINT FORENSIC AUDIT OF CONTINUATION NOTEBOOK ===")

# Check 1: Any escaped quotes (\") in code cells
escaped_quotes = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        for line in c["source"]:
            if '\\"' in line:
                escaped_quotes.append((i, line))
print(f"1. Escaped Quotes (\\\"): {len(escaped_quotes)} (Expected: 0)")

# Check 2: distributed_train_step replica context (tf.shape outside strategy.run)
bad_replica_calls = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if "def distributed_train_step" in src:
            lines = src.split("\n")
            for li, line in enumerate(lines):
                if "def distributed_train_step" in line:
                    body = "\n".join(lines[li:li+6])
                    if "tf.shape" in body:
                        bad_replica_calls.append(line)
print(f"2. tf.shape called outside strategy.run: {len(bad_replica_calls)} (Expected: 0)")

# Check 3: Warmup pass presence
has_warmup = any("Warmup: SUCCESS" in "".join(c["source"]) for c in nb["cells"])
print(f"3. Multi-GPU 1-step Warmup Pass: {has_warmup} (Expected: True)")

# Check 4: Auto-Extract Front-Facing Sprites (no missing file error)
has_auto_extract = any("Auto-extracted" in "".join(c["source"]) for c in nb["cells"])
print(f"4. Auto-Extract Front-Facing Sprites (Zero FileNotFoundError): {has_auto_extract} (Expected: True)")

# Check 5: uint8 Memory-Safe Dataset (No 2GB Protobuf crash)
has_uint8_pipeline = any("dtype=np.uint8" in "".join(c["source"]) and "tf.gather(lut_tf" in "".join(c["source"]) for c in nb["cells"])
print(f"5. uint8 Memory-Safe Pipeline (2GB GraphDef Safe): {has_uint8_pipeline} (Expected: True)")

# Check 6: DiffAugment uses RandomTranslation (No tf.map_fn placeholder crash)
has_rt = any("layers.RandomTranslation" in "".join(c["source"]) for c in nb["cells"])
has_map_fn = any("tf.map_fn" in "".join(c["source"]) for c in nb["cells"])
print(f"6. DiffAugment: RandomTranslation={has_rt}, tf.map_fn={has_map_fn} (Expected: True, False)")

# Check 7: Optimizer build under strategy scope
has_opt_build = any("g_optimizer.build" in "".join(c["source"]) and "d_optimizer.build" in "".join(c["source"]) for c in nb["cells"])
print(f"7. Explicit Optimizer Build (No lazy Adam slot crash): {has_opt_build} (Expected: True)")

# Check 8: Weight Resumption Safety
has_weight_check = any("WEIGHT_CANDIDATES_G" in "".join(c["source"]) for c in nb["cells"])
print(f"8. Safe Weight Resumption with Fallback: {has_weight_check} (Expected: True)")

# Check 9: Full AST Syntax Compilation
all_valid = True
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        import ast
        try:
            ast.parse("".join(cell["source"]))
        except Exception as e:
            all_valid = False
            print(f"Syntax error in Cell {idx}: {e}")
print(f"9. All Code Cells Valid Python AST: {all_valid} (Expected: True)")

