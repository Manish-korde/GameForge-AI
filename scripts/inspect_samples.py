import json
with open('backend/alucard_samples/manifest.json') as f:
    manifest = json.load(f)
for x in manifest:
    print(f"{x['sample_id']}: {x['filename']} - {x['label']} | {x['text_prompt']}")
