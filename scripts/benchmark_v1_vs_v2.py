"""
Comprehensive Side-by-Side Comparative Audit:
Model A (Baseline): models/4_Transformer_2k_v1
Model B (New 50-Epoch Repaired): C:\\Users\\manis\\Downloads\\gameforge_flan_t5_small_v2
"""

import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PATH_V1 = os.path.abspath("models/4_Transformer_2k_v1")
PATH_V2 = r"C:\Users\manis\Downloads\gameforge_flan_t5_small_v2"
TASK_PREFIX = "game concept: "
REQUIRED_FIELDS = {"theme", "environment", "characters", "weapons", "props", "visualStyle"}

def main():
    print("=" * 70)
    print("⚔️ SIDE-BY-SIDE BENCHMARK: MODEL V1 (BASELINE) vs MODEL V2 (RETRAINED)")
    print("=" * 70)

    print(f"Loading Model V1 from: {PATH_V1}...")
    tok_v1 = AutoTokenizer.from_pretrained(PATH_V1)
    mod_v1 = AutoModelForSeq2SeqLM.from_pretrained(PATH_V1)
    mod_v1.eval()

    print(f"Loading Model V2 from: {PATH_V2}...")
    tok_v2 = AutoTokenizer.from_pretrained(PATH_V2)
    mod_v2 = AutoModelForSeq2SeqLM.from_pretrained(PATH_V2)
    mod_v2.eval()
    print("Both models loaded successfully!\n")

    # 1. Exact Failure Mode Test: "candy farming game with a fox"
    test_prompt = "candy farming game with a fox"
    print("=" * 70)
    print(f"🧪 TEST 1: THE FAILING SCREENSHOT PROMPT: '{test_prompt}'")
    print("=" * 70)

    def generate_single(model, tokenizer, prompt_str):
        inputs = tokenizer(f"{TASK_PREFIX}{prompt_str}", return_tensors="pt", truncation=True, max_length=256)
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=1,
                do_sample=False,
                max_new_tokens=384
            )
        lat_ms = round((time.perf_counter() - start) * 1000, 1)
        raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        try:
            parsed = json.loads(raw_text)
            valid = True
        except Exception:
            parsed = None
            valid = False
        return raw_text, parsed, valid, lat_ms

    raw1, p1, v1, lat1 = generate_single(mod_v1, tok_v1, test_prompt)
    raw2, p2, v2, lat2 = generate_single(mod_v2, tok_v2, test_prompt)

    print("--- [MODEL V1: BASELINE OUTPUT] ---")
    print(f"Latency: {lat1} ms | Valid JSON: {v1}")
    if p1:
        print(f"Theme:       {p1.get('theme')}")
        print(f"Environment: {p1.get('environment')}")
        print(f"Characters:  {[c.get('name') for c in p1.get('characters', [])]}")
        print(f"Weapons:     {[w.get('name') for w in p1.get('weapons', [])]}")
        print(f"Props:       {[p.get('name') for p in p1.get('props', [])]} (Count: {len(p1.get('props', []))})")
    else:
        print("Raw output:", raw1)

    print("\n--- [MODEL V2: RETRAINED 50-EPOCH OUTPUT] ---")
    print(f"Latency: {lat2} ms | Valid JSON: {v2}")
    if p2:
        print(f"Theme:       {p2.get('theme')}")
        print(f"Environment: {p2.get('environment')}")
        print(f"Characters:  {[c.get('name') for c in p2.get('characters', [])]}")
        print(f"Weapons:     {[w.get('name') for w in p2.get('weapons', [])]}")
        print(f"Props:       {[p.get('name') for p in p2.get('props', [])]} (Count: {len(p2.get('props', []))})")
        print("\nFull V2 JSON:\n", json.dumps(p2, indent=2))
    else:
        print("Raw output:", raw2)

    # 2. Generalization Test across 5 Distinct Game Archetypes
    archetype_prompts = [
        "cyberpunk hacker infiltrating corporate data tower",
        "cozy mountain cabin foraging sim with a companion dog",
        "western sheriff in dusty frontier town hunting bandits",
        "space exploration with robot companion searching for energy cells",
        "dark gothic castle with an undead knight searching for an ancient relic"
    ]

    print("\n" + "=" * 70)
    print("🧪 TEST 2: MULTI-GENRE GENERALIZATION COMPARISON")
    print("=" * 70)

    for ap in archetype_prompts:
        print(f"\n▶ Prompt: '{ap}'")
        _, out1, val1, _ = generate_single(mod_v1, tok_v1, ap)
        _, out2, val2, _ = generate_single(mod_v2, tok_v2, ap)

        p1_props = len(out1.get("props", [])) if out1 else 0
        p2_props = len(out2.get("props", [])) if out2 else 0

        p1_chars = [c.get("name") for c in out1.get("characters", [])] if out1 else []
        p2_chars = [c.get("name") for c in out2.get("characters", [])] if out2 else []

        p1_weap = [w.get("name") for w in out1.get("weapons", [])] if out1 else []
        p2_weap = [w.get("name") for w in out2.get("weapons", [])] if out2 else []

        print(f"  [Model V1] Theme: {out1.get('theme') if out1 else 'Err'} | Chars: {p1_chars} | Weap: {p1_weap} | Props ({p1_props}): {[p.get('name') for p in out1.get('props', [])] if out1 else []}")
        print(f"  [Model V2] Theme: {out2.get('theme') if out2 else 'Err'} | Chars: {p2_chars} | Weap: {p2_weap} | Props ({p2_props}): {[p.get('name') for p in out2.get('props', [])] if out2 else []}")

    # 3. Statistical Test over 50 held-out samples from test_repaired.jsonl
    test_path = "transformer_training/test_repaired.jsonl"
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            test_rows = [json.loads(line) for line in f if line.strip()][:50]
        
        print("\n" + "=" * 70)
        print(f"📊 TEST 3: RIGOROUS EVALUATION OVER {len(test_rows)} HELD-OUT TEST BLUEPRINTS")
        print("=" * 70)

        stats_v1 = {"valid_json": 0, "full_schema": 0, "non_empty_props": 0, "latency_total": 0}
        stats_v2 = {"valid_json": 0, "full_schema": 0, "non_empty_props": 0, "latency_total": 0}

        for row in test_rows:
            p_text = row["input"]
            _, o1, is_v1, l1 = generate_single(mod_v1, tok_v1, p_text)
            _, o2, is_v2, l2 = generate_single(mod_v2, tok_v2, p_text)

            stats_v1["latency_total"] += l1
            stats_v2["latency_total"] += l2

            if is_v1 and o1:
                stats_v1["valid_json"] += 1
                if REQUIRED_FIELDS.issubset(set(o1.keys())):
                    stats_v1["full_schema"] += 1
                if o1.get("props") and len(o1["props"]) > 0:
                    stats_v1["non_empty_props"] += 1

            if is_v2 and o2:
                stats_v2["valid_json"] += 1
                if REQUIRED_FIELDS.issubset(set(o2.keys())):
                    stats_v2["full_schema"] += 1
                if o2.get("props") and len(o2["props"]) > 0:
                    stats_v2["non_empty_props"] += 1

        n = len(test_rows)
        print(f"\n==================== FINAL AUDIT COMPARISON MATRIX ({n} Samples) ====================")
        print(f"Metric                        | Model V1 (Baseline)  | Model V2 (Retrained 50-Epoch) | Improvement")
        print(f"-----------------------------------------------------------------------------------------")
        print(f"Syntactic JSON Parse Rate     | {stats_v1['valid_json']/n*100:6.1f}% ({stats_v1['valid_json']}/{n})   | {stats_v2['valid_json']/n*100:6.1f}% ({stats_v2['valid_json']}/{n})      | {stats_v2['valid_json']/n*100 - stats_v1['valid_json']/n*100:+5.1f}%")
        print(f"Full Schema Completeness Rate | {stats_v1['full_schema']/n*100:6.1f}% ({stats_v1['full_schema']}/{n})   | {stats_v2['full_schema']/n*100:6.1f}% ({stats_v2['full_schema']}/{n})      | {stats_v2['full_schema']/n*100 - stats_v1['full_schema']/n*100:+5.1f}%")
        print(f"Non-Empty Props Rate (Target) | {stats_v1['non_empty_props']/n*100:6.1f}% ({stats_v1['non_empty_props']}/{n})   | {stats_v2['non_empty_props']/n*100:6.1f}% ({stats_v2['non_empty_props']}/{n})      | {stats_v2['non_empty_props']/n*100 - stats_v1['non_empty_props']/n*100:+5.1f}%")
        print(f"Avg CPU Inference Latency     | {stats_v1['latency_total']/n:6.1f} ms          | {stats_v2['latency_total']/n:6.1f} ms             | {stats_v2['latency_total']/n - stats_v1['latency_total']/n:+5.1f} ms")
        print("=" * 85)

if __name__ == "__main__":
    main()
