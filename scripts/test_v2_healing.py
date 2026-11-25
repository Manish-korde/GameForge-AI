import re, json

def repair_v2_syntax(raw: str):
    cleaned = raw.strip()
    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned
    if not cleaned.endswith("}"):
        cleaned = cleaned + "}"
    
    # Check if raw valid
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    
    # Fix ["name": "...", "desc": "..."] -> [{"name": "...", "desc": "..."}]
    def fix_list_brackets(field_name, text):
        pattern = rf'"{field_name}"\s*:\s*\[(.*?)\]'
        m = re.search(pattern, text)
        if not m: return text
        inner = m.group(1)
        # Find all name and desc pairs
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', inner)
        descs = re.findall(r'"desc"\s*:\s*"([^"]+)"', inner)
        objs = [{"name": n, "desc": d} for n, d in zip(names, descs)]
        rep = f'"{field_name}": ' + json.dumps(objs)
        return re.sub(pattern, rep, text, count=1)
    
    repaired = cleaned
    for f in ["characters", "weapons", "props"]:
        repaired = fix_list_brackets(f, repaired)
        
    try:
        return json.loads(repaired)
    except Exception:
        pass
    
    # Fallback regex extraction
    theme_m = re.search(r'"theme"\s*:\s*"([^"]+)"', raw)
    env_m = re.search(r'"environment"\s*:\s*"([^"]+)"', raw)
    style_m = re.search(r'"visualStyle"\s*:\s*"([^"]+)"', raw)
    
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', raw)
    descs = re.findall(r'"desc"\s*:\s*"([^"]+)"', raw)
    return {
        "theme": theme_m.group(1) if theme_m else "Candy Farming",
        "environment": env_m.group(1) if env_m else "Candy Cane Forest",
        "characters": [{"name": names[0], "desc": descs[0]}] if names else [{"name": "Hero", "desc": "A traveler."}],
        "weapons": [{"name": names[1], "desc": descs[1]}] if len(names) > 1 else [{"name": "Tool", "desc": "A tool."}],
        "props": [{"name": names[2], "desc": descs[2]}] if len(names) > 2 else [{"name": "Prop", "desc": "An item."}],
        "visualStyle": style_m.group(1) if style_m else "Stylized 2D"
    }

raw_test = '"theme": "Candy Farming", "environment": "Spicious Candy Cane Forest", "characters": ["name": "Sniper", "desc": "A cheerful farmer cultivating sugar beets and candy cane groves.", "name": "Fox Harvester", "desc": "A cheerful farmer cultivating sugar beets and candy cane groves."], "weapons": ["name": "Spicy Cane Sprout", "desc": "A sturdy straw straw that drains the life of its harvest."], "props": ["name": "Candy Cane Sprout", "desc": "An interactive agricultural shoot that yields sweet peppermint syrup and caramel syrup."], "visualStyle": "Stylized 2D"'
repaired_obj = repair_v2_syntax(raw_test)
print("Repaired Result:")
print(json.dumps(repaired_obj, indent=2))
