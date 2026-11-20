import json
import zlib
import base64
import os

with open('alucard_frontfacing_42k_indices.json', 'r', encoding='utf-8') as f:
    indices = json.load(f)

compressed = zlib.compress(json.dumps(indices).encode('utf-8'), level=9)
b64_str = base64.b64encode(compressed).decode('ascii')

with open('scripts/build_discrete_palette_dcgan_notebook.py', 'r', encoding='utf-8') as f:
    script = f.read()

target = '''# Hard failure safeguard: Do not silently fallback to arbitrary unaligned data
if not curated_indices:
    raise FileNotFoundError(
        f"Front-facing indices file '{INDICES_FILE_NAME}' not found in candidate paths. "
        "Upload alucard_frontfacing_42k_indices.json or final_alucard_frontfacing_manifest.csv as a Kaggle dataset."
    )'''

replacement = f'''# Self-contained fallback: Unpack pre-embedded 42,212 front-facing indices if not found on disk
if not curated_indices:
    import zlib, base64
    print("Candidate files not found on disk. Unpacking pre-embedded 42,212 front-facing indices...")
    EMBEDDED_B64 = "{b64_str}"
    curated_indices = json.loads(zlib.decompress(base64.b64decode(EMBEDDED_B64.encode('ascii'))).decode('utf-8'))
    print(f"Successfully unpacked {{len(curated_indices):,}} embedded front-facing indices!")'''

assert target in script, "Target string not found in script!"
script_new = script.replace(target, replacement)

with open('scripts/build_discrete_palette_dcgan_notebook.py', 'w', encoding='utf-8') as f:
    f.write(script_new)

print("Successfully updated build script!")

