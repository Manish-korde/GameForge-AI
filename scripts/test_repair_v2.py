import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from backend.transformer_service import repair_json_syntax

# Raw output from Model V2 for "candy farming game with a fox"
raw = '"theme": "Candy Farming", "environment": "Spicious Candy Cane Forest", "characters": ["name": "Sniper", "desc": "A cheerful farmer cultivating sugar beets and candy cane groves.", "name": "Fox Harvester", "desc": "A cheerful farmer cultivating sugar beets and candy cane groves."], "weapons": ["name": "Spicy Cane Sprout", "desc": "A sturdy straw straw that drains the life of its harvest."], "props": ["name": "Candy Cane Sprout", "desc": "An interactive agricultural shoot that yields sweet peppermint syrup and caramel syrup.", "name": "Candy Cane Sprout", "desc": "An interactive agricultural shoot that yields sweet peppermint syrup and caramel syrup."], "visualStyle": "Stylized 2D"'

healed = repair_json_syntax("{" + raw + "}")
print("Healed direct:", healed)
if not healed:
    # Let's inspect regex extraction
    import re
    theme_m = re.search(r'"theme"\s*:\s*"([^"]+)"', raw)
    env_m = re.search(r'"environment"\s*:\s*"([^"]+)"', raw)
    style_m = re.search(r'"visualStyle"\s*:\s*"([^"]+)"', raw)
    
    char_chunk = re.search(r'"characters"\s*:\s*\[(.*?)\](?=,\s*"weapons")', raw)
    weap_chunk = re.search(r'"weapons"\s*:\s*\[(.*?)\](?=,\s*"props")', raw)
    prop_chunk = re.search(r'"props"\s*:\s*\[(.*?)\](?=,\s*"visualStyle")', raw)
    
    def extract_pairs(chunk):
        if not chunk: return []
        txt = chunk.group(1)
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', txt)
        descs = re.findall(r'"desc"\s*:\s*"([^"]+)"', txt)
        return [{"name": n, "desc": d} for n, d in zip(names, descs)]

    extracted = {
        "theme": theme_m.group(1) if theme_m else "Candy Farming",
        "environment": env_m.group(1) if env_m else "Candy Cane Forest",
        "characters": extract_pairs(char_chunk),
        "weapons": extract_pairs(weap_chunk),
        "props": extract_pairs(prop_chunk),
        "visualStyle": style_m.group(1) if style_m else "Stylized 2D"
    }
    print("Regex extracted:")
    print(json.dumps(extracted, indent=2))
