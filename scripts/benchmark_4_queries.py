import os
import json
import numpy as np

os.environ['HF_DATASETS_OFFLINE'] = '1'
from datasets import load_dataset

index_path = 'models/latent_gallery_index_25k.npy'
manifest_path = 'models/latent_gallery_manifest_25k.json'
gallery = np.load(index_path)
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

# Query 1: Sword (id: 200319)
# Query 2: Potion (id: 200081)
# Query 3: Enemy (id: 200001)
# Query 4: Character (id: 200000)

queries = [
    ("Sword (Weapon)", 200319),
    ("Potion (Item)", 200081),
    ("Monster (Enemy)", 200001),
    ("Warrior (Character)", 200000)
]

print("=== VAE RETRIEVAL BENCHMARK: 4 DIVERSE QUERIES (TOP-5) ===")

for name, q_id in queries:
    q_idx = q_id - 200000
    q_vec = gallery[q_idx]
    q_cat = manifest[q_idx]['category']
    q_text = manifest[q_idx]['text']
    
    # Compute cosine similarity across all 25,000 assets
    sims = np.dot(gallery, q_vec)
    # Exclude self
    sims[q_idx] = -1.0
    
    top5_indices = np.argsort(sims)[::-1][:5]
    
    print(f"\n--- QUERY: {name} (ID: {q_id}) ---")
    print(f"Ground Truth Category: {q_cat} | Prompt: '{q_text}'")
    same_cat_count = 0
    for rank, idx in enumerate(top5_indices, 1):
        m = manifest[idx]
        sim = float(sims[idx])
        is_same = (m['category'].lower() == q_cat.lower())
        if is_same: same_cat_count += 1
        tag = "[MATCH]" if is_same else "[DIFF] "
        print(f"  Rank {rank}: {tag} Sim={sim:.4f} | ID={m['id']} | Category={m['category']:10s} | Label={m['label']:20s} | '{m['text'][:40]}...'")
    p5 = same_cat_count / 5.0
    print(f"Precision@5: {p5*100:.1f}%")
