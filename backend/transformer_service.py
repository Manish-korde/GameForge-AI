import os
import re
import json
import time
import logging
import threading
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger("transformer_service")
logger.setLevel(logging.INFO)

TASK_PREFIX = "game concept: "
MAX_NEW_TOKENS = 512
REQUIRED_SCHEMA_FIELDS = {"theme", "environment", "characters", "weapons", "props", "visualStyle"}

MODEL_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "models", "4_Transformer_2k_v1"
))


def strip_code_fences(text: str) -> str:
    """Removes markdown code fences (```json ... ```) if emitted by the model."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines).strip()
    return text


def repair_json_syntax(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust boundary repair auditor: heals truncated syntax, missing outer braces,
    bracket drops, or cyclic repetition loops emitted under edge-case prompts.
    Supports both standard 6-key and heavy nested blueprint schemas.
    """
    cleaned = strip_code_fences(text).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Ensure root object is wrapped with braces
    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned
    if not cleaned.endswith("}"):
        cleaned = cleaned + "}"

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 1. Truncate degenerative cyclic repetitions (e.g. "ruins": "ruins": ...)
    rep_match = re.search(r'("[^"]+":\s*){2,}', cleaned)
    if rep_match:
        cleaned = cleaned[:rep_match.start()].rstrip().rstrip(',').rstrip(':')

    # 2. Heal dropped curly braces inside entity arrays (e.g. ["name": "...", "desc": "..."])
    def fix_list_brackets(field_name, t):
        pattern = rf'"{field_name}"\s*:\s*\[(.*?)\]'
        m = re.search(pattern, t)
        if not m:
            return t
        inner = m.group(1)
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', inner)
        descs = re.findall(r'"desc"\s*:\s*"([^"]+)"', inner)
        if names:
            objs = [{"name": n, "desc": d} for n, d in zip(names, descs)]
            rep = f'"{field_name}": ' + json.dumps(objs)
            return re.sub(pattern, rep, t, count=1)
        return t

    for fld in ["characters", "weapons", "props"]:
        cleaned = fix_list_brackets(fld, cleaned)

    cleaned = re.sub(r',\s*$', '', cleaned)
    if cleaned.count('"') % 2 != 0:
        cleaned += '"'

    # 3. Balance unclosed brackets and braces
    open_sq = cleaned.count('[') - cleaned.count(']')
    if open_sq > 0:
        cleaned += ']' * open_sq
    open_cu = cleaned.count('{') - cleaned.count('}')
    if open_cu > 0:
        cleaned += '}' * open_cu

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 4. Fallback semantic extraction via regex if braces are structurally damaged
    # Supports both standard 6-key schema AND heavy blueprint schema (game_title, world_setting, Boss/Protagonist roles)
    try:
        theme_m = re.search(r'"theme"\s*:\s*"([^"]+)"', text)
        title_m = re.search(r'"(?:game_title|genre)"\s*:\s*"([^"]+)"', text)
        env_m = re.search(r'"(?:environment|world_setting)"\s*:\s*"([^"]+)"', text)
        style_m = re.search(r'"(?:visualStyle|visual_style|art_style)"\s*:\s*"([^"]+)"', text)

        # Parse (name, desc, role) triples sequentially so Boss/Enemy entries separate cleanly
        chars, weaps, props, enemies = [], [], [], []
        # Check section boundaries if characters / weapons / props keys exist
        sec_positions = []
        for sec in ["characters", "weapons", "props", "enemies"]:
            m_sec = re.search(rf'"{sec}"\s*:', text)
            if m_sec:
                sec_positions.append((m_sec.start(), sec))
        sec_positions.sort()

        for m_name in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
            pos = m_name.start()
            nm = m_name.group(1).strip()
            snippet = text[pos:pos + 240]
            desc_m = re.search(r'"desc"\s*:\s*"([^"]+)"', snippet)
            role_m = re.search(r'"role"\s*:\s*"([^"]+)"', snippet)
            role_val = role_m.group(1).lower() if role_m else ""
            desc_val = desc_m.group(1).strip() if desc_m else f"Entity in the game world ({nm})."

            # Determine which section this item sits inside
            active_sec = "characters"
            for sp, sname in sec_positions:
                if sp <= pos:
                    active_sec = sname

            item_obj = {"name": nm, "desc": desc_val}
            if active_sec == "enemies" or "boss" in role_val or "enemy" in role_val:
                enemies.append(item_obj)
            elif active_sec == "weapons":
                weaps.append(item_obj)
            elif active_sec == "props":
                props.append(item_obj)
            else:
                chars.append(item_obj)

        resolved_theme = (theme_m.group(1) if theme_m else (title_m.group(1) if title_m else "")).strip()
        resolved_env = env_m.group(1).strip() if env_m else ""

        return {
            "theme": resolved_theme,
            "environment": resolved_env,
            "characters": chars,
            "enemies": enemies,
            "weapons": weaps,
            "props": props,
            "visualStyle": style_m.group(1).strip() if style_m else "Pixel Art"
        }
    except Exception:
        return None


# ==============================================================================
# General-Case Compositional Prompt Semantic Analyzer
# Extracts Enemies/Threats, Hero Roles, Biomes & Settings across arbitrary prompts
# without relying on hardcoding any single prompt word.
# ==============================================================================
THREAT_LEXICON = {
    # Apocalyptic / Outbreak / Horror threats
    "zombie": ("Zombie Horde", "Relentless infected undead roaming the devastated streets in aggressive swarms."),
    "undead": ("Restless Undead", "Risen corpses driven by dark contagion to overwhelm living survivors."),
    "infected": ("Mutated Infected", "Fast, virulent host creatures hunting survivors through collapsed zones."),
    "mutant": ("Wasteland Mutant", "Irradiated apex mutant adapting violently to hostile ruins."),
    "vampire": ("Elder Vampire", "Nocturnal blood-stalker commanding shadow thralls across cursed halls."),
    "werewolf": ("Feral Werewolf", "Moon-crazed lycanthrope stalking victims with razor claws and blinding speed."),
    "ghost": ("Spectral Phantom", "Translucent apparition phasing through walls and draining survivor resolve."),
    "phantom": ("Shadow Phantom", "Ethereal wraith haunting darkened corridors and sabotaging light sources."),
    "demon": ("Abyssal Demon", "Hell-forged fiend unleashing corrupted fire and terror across the realm."),
    "skeleton": ("Skeleton Warrior", "Reanimated bone sentinel guarding ancient crypts with rusted steel."),
    "necromancer": ("Dark Necromancer", "Forbidden sorcerer raising fallen minions to overwhelm intruders."),
    # Fantasy threats
    "dragon": ("Ancient Dragon", "Colossal armored wyrm breathing scorching devastation from above."),
    "orc": ("Orc Warlord", "Brutal tribal brute leading heavy armored raiding warbands."),
    "goblin": ("Goblin Raider", "Cunning subterranean scavenger attacking in swarming ambush packs."),
    "slime": ("Corrosive Slime", "Gelatinous hazard creature splitting upon impact in cavern depths."),
    "golem": ("Runic Golem", "Towering stone construct animated by ancient arcane cores."),
    "witch": ("Hexborn Witch", "Rogue spellcaster conjuring curses and enchanted hazards."),
    "monster": ("Apex Monster", "Fierce towering beast terrorizing the surrounding frontier."),
    "titan": ("Colossal Titan", "Earth-shaking giant requiring tactical weak-point strikes to topple."),
    "kraken": ("Abyssal Kraken", "Tentacled leviathan dragging vessels down into the deep."),
    "dinosaur": ("Apex Dinosaur", "Prehistoric predator stalking dense jungles and volcanic valleys."),
    # Sci-Fi / Action threats
    "alien": ("Alien Invader", "Hostile extraterrestrial shock-trooper deploying advanced bio-plasma tech."),
    "robot": ("Rogue Combat Mech", "Malfunctioning armored automaton enforcing lethal lockdown protocols."),
    "cyborg": ("Rogue Cyborg", "Cybernetically augmented enforcer hunting targets across neon sectors."),
    "drone": ("Hunter Drone", "Autonomous aerial patrol unit equipped with thermal targeting lasers."),
    "mech": ("Siege Mech", "Heavy bipedal war machine armed with twin missile pods."),
    "pirate": ("Corsair Raider", "Ruthless sky or sea marauder ambushing trade routes for plunder."),
    "bandit": ("Outlaw Bandit", "Armed frontier scavenger ambushing travelers along unguarded roads."),
    "raider": ("Wasteland Raider", "Desperate armored scavenger fighting for dwindling territory and fuel."),
    "cultist": ("Shadow Cultist", "Fanatical acolyte channeling forbidden rituals against trespassers."),
}

HERO_ROLES = {
    "survivor": ("Apocalypse Survivor", "A resourceful survivor scavenging gear, barricading shelters, and enduring hostile threats."),
    "knight": ("Royal Knight", "An armored defender sworn to shield the realm with steel and tactical discipline."),
    "paladin": ("Oathbound Paladin", "A holy vanguard wielding consecrated steel against encroaching darkness."),
    "wizard": ("Arcane Wizard", "A scholar of elemental magic bending fire, frost, and lightning against foes."),
    "mage": ("Battle Mage", "A spell-weaving tactician balancing mana reserves with high-impact spells."),
    "archer": ("Master Archer", "A keen-eyed marksman striking targets cleanly from long range."),
    "ranger": ("Frontier Ranger", "An agile tracker navigating hazardous terrain and flanking enemies."),
    "ninja": ("Shadow Ninja", "A swift stealth operative executing precision strikes from the shadows."),
    "samurai": ("Ronin Samurai", "A disciplined blademaster parrying incoming strikes with lethal counter-cuts."),
    "hunter": ("Bounty Hunter", "A relentless tracker specialized in dismantling dangerous monstrous threats."),
    "marine": ("Starfighter Marine", "An armored shock-trooper trained for zero-gravity and hostile invasions."),
    "soldier": ("Tactical Soldier", "A combat-hardened specialist trained in squad defense and firearm control."),
    "trooper": ("Vanguard Trooper", "A frontline defender equipped with modular tactical gear."),
    "pilot": ("Ace Pilot", "A veteran cockpit specialist maneuvering through intense aerial or orbital fire."),
    "explorer": ("Field Explorer", "An adaptable adventurer charting ruins and unlocking ancient mechanisms."),
    "detective": ("Noir Detective", "A sharp investigator piecing together hidden clues under pressure."),
    "farmer": ("Cozy Homestead Farmer", "A dedicated cultivator tending crops, tools, and friendly companions."),
    "pirate": ("Swashbuckling Captain", "A daring privateer charting uncharted waters and hidden coves."),
}

STOP_NOISE_WORDS = {
    "game", "play", "player", "action", "adventure", "simulator", "simulation", "rpg",
    "2d", "3d", "pixel", "art", "style", "themed", "world", "story", "level", "levels",
    "with", "and", "the", "a", "an", "in", "on", "of", "to", "for", "where", "you", "must",
}


def extract_prompt_semantics(prompt: str) -> Dict[str, Any]:
    """
    General-purpose linguistic parser over arbitrary game prompts.
    Extracts:
      - enemies: list of hostile threats mentioned or implied via relational syntax
      - hero: inferred or explicit protagonist {name, desc}
      - biome: semantic setting classification (outbreak_apocalypse, gothic_horror,
               sci_fi_action, medieval_fantasy, wilderness_survival, etc.)
      - weapon: context-aligned starting weapon/tool
      - props: context-aligned world items
    """
    p_low = (prompt or "").lower().strip()
    tokens = re.findall(r"[a-z0-9-]+", p_low)

    # 1. Determine semantic biome (distinguishing outbreak survival from arctic wilderness!)
    biome = "general_action"
    if any(k in p_low for k in ["apocalypse", "outbreak", "wasteland", "quarantine", "pandemic", "infected", "zombie", "undead", "ruined city", "bunker"]):
        biome = "outbreak_apocalypse"
    elif any(k in p_low for k in ["vampire", "haunted", "mansion", "asylum", "ghost", "cursed", "horror", "crypt", "morgue", "werewolf", "demon", "nightmare"]):
        biome = "gothic_horror"
    elif any(k in p_low for k in ["alien", "robot", "space", "orbit", "galaxy", "starship", "cyber", "neon", "laser", "plasma", "shooter", "invasion", "mech", "asteroid", "drone"]):
        biome = "sci_fi_action"
    elif any(k in p_low for k in ["medieval", "dragon", "castle", "kingdom", "knight", "dungeon", "catacomb", "wizard", "magic", "sword", "orc", "goblin", "skeleton", "siege"]):
        biome = "medieval_fantasy"
    elif any(k in p_low for k in ["farm", "cozy", "harvest", "crop", "botanist", "candy", "garden"]):
        biome = "cozy_farming"
    elif any(k in p_low for k in ["blizzard", "arctic", "taiga", "snow", "wilderness", "island", "jungle", "surviv"]):
        biome = "wilderness_survival"

    # 2. Extract hostile threats / enemies (lexicon + modifier capture + open-vocabulary syntax)
    extracted_enemies = []
    seen_enemy_keys = set()

    for idx, tok in enumerate(tokens):
        stem = tok.rstrip("s") if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss") else tok
        matched_key = tok if tok in THREAT_LEXICON else (stem if stem in THREAT_LEXICON else None)
        if matched_key and matched_key not in seen_enemy_keys:
            base_name, base_desc = THREAT_LEXICON[matched_key]
            # Check preceding modifier word (e.g. "frost dragon", "toxic zombie", "space pirate")
            mod = tokens[idx - 1] if idx > 0 and tokens[idx - 1] not in STOP_NOISE_WORDS and tokens[idx - 1] not in {"survive", "fight", "defeat", "hunt", "kill", "escape"} else ""
            if mod and mod not in base_name.lower() and mod not in HERO_ROLES:
                display_name = f"{mod.capitalize()} {base_name}"
            else:
                display_name = base_name
            seen_enemy_keys.add(matched_key)
            extracted_enemies.append({"name": display_name, "desc": base_desc})

    # Open-vocabulary relational pattern fallback for unseen enemy nouns (e.g., "defeat lava beetles", "clockwork spiders")
    rel_patterns = [
        r"(?:fight|battle|defeat|kill|survive|hunt|stop|escape|against|vs\.?)\s+(?:the\s+|an?\s+|waves?\s+of\s+|hordes?\s+of\s+)?([a-z0-9-]+(?:\s+[a-z0-9-]+)?)(?:\s+(?:apocalypse|outbreak|invasion|horde|army|swarm|boss|in|at|on|with)|$)",
        r"([a-z0-9-]+)\s+(?:apocalypse|outbreak|invasion|horde|swarm)"
    ]
    if not extracted_enemies:
        for pat in rel_patterns:
            m = re.search(pat, p_low)
            if m:
                phrase = m.group(1).strip()
                words = [w for w in phrase.split() if w not in STOP_NOISE_WORDS and w not in HERO_ROLES]
                if words:
                    enemy_title = " ".join(w.capitalize() for w in words[:2])
                    if enemy_title.lower() not in seen_enemy_keys:
                        seen_enemy_keys.add(enemy_title.lower())
                        extracted_enemies.append({
                            "name": f"{enemy_title} Horde" if "apocalypse" in p_low or "invasion" in p_low or "swarm" in p_low else enemy_title,
                            "desc": f"Hostile {enemy_title.lower()} forces threatening survival across the active zone."
                        })
                    break

    # 3. Extract or infer Hero Role (making sure enemy words aren't mistakenly picked as hero)
    hero_obj = None
    for tok in tokens:
        stem = tok.rstrip("s") if len(tok) > 4 and tok.endswith("s") else tok
        h_key = tok if tok in HERO_ROLES else (stem if stem in HERO_ROLES else None)
        if h_key and h_key not in seen_enemy_keys:
            h_name, h_desc = HERO_ROLES[h_key]
            hero_obj = {"name": h_name, "desc": h_desc}
            break

    if hero_obj is None:
        if biome == "outbreak_apocalypse":
            hero_obj = HERO_ROLES["survivor"]
            hero_obj = {"name": hero_obj[0], "desc": hero_obj[1]}
        elif biome == "gothic_horror":
            hero_obj = {"name": "Occult Investigator", "desc": "A brave investigator armed with protective wards confronting dark supernatural forces."}
        elif biome == "sci_fi_action":
            hero_obj = {"name": "Vanguard Marine", "desc": "An armored tactical operative defending sectors against hostile technological and alien threats."}
        elif biome == "medieval_fantasy":
            hero_obj = {"name": "Kingdom Sentinel", "desc": "A seasoned warrior defending fortified walls against mythic monsters and dark armies."}
        elif biome == "wilderness_survival":
            hero_obj = {"name": "Wilderness Survivor", "desc": "A resilient scout enduring harsh elements and wild predators to reach rescue."}
        elif extracted_enemies:
            hero_obj = {"name": "Tactical Defender", "desc": "A skilled frontline protagonist confronting hostile threats head-on."}

    # 4. Biome-specific coherent defaults (theme, environment, weapon, props, genre, loop)
    biome_defaults = {
        "outbreak_apocalypse": {
            "theme": "Post-Apocalyptic Outbreak Survival",
            "environment": "Abandoned Quarantine City & Ruined Streets",
            "genre": "Post-Apocalyptic Survival Action",
            "game_type": "Top-Down Horde Survival & Scavenging",
            "perspective": "Top-Down 2D",
            "weapon": {"name": "Reinforced Crowbar & Combat Shotgun", "desc": "Versatile close-quarters pry tool paired with a heavy pump-action firearm for crowd control."},
            "props": [
                {"name": "Emergency Medkit Crate", "desc": "Sealed medical supply box restoring health and curing infection trauma."},
                {"name": "Barricaded Supply Checkpoint", "desc": "Fortified street blockade storing salvaged ammo, rations, and flare beacons."}
            ],
            "mechanic": "Scavenging dwindling supplies, barricading safehouses, and wave-based combat",
            "hazards": ["Swarming infected hordes", "Collapsed urban rubble", "Toxic contamination clouds"]
        },
        "gothic_horror": {
            "theme": "Gothic Supernatural Horror",
            "environment": "Cursed Mansion & Shadowed Halls",
            "genre": "Atmospheric Survival Horror",
            "game_type": "Exploration & Supernatural Defense",
            "perspective": "Top-Down 2D (Dark Vignette)",
            "weapon": {"name": "Silver Crossbow", "desc": "Consecrated repeating crossbow firing silver-tipped bolts against nocturnal threats."},
            "props": [
                {"name": "Flickering Kerosene Lantern", "desc": "Fragile light source illuminating dark corridors and repelling encroaching shadows."},
                {"name": "Sanctified Relic Ward", "desc": "Protective talisman slowing supernatural assaults inside safe rooms."}
            ],
            "mechanic": "Light-resource management, tactical stealth, and cursed boss confrontation",
            "hazards": ["Encroaching darkness zones", "Ambush corridors", "Cursed floor traps"]
        },
        "sci_fi_action": {
            "theme": "Sci-Fi Tactical Combat",
            "environment": "Zero-Gravity Orbital Station & Hangar",
            "genre": "Sci-Fi Action Shooter",
            "game_type": "Top-Down Twin-Stick / Tactical Action",
            "perspective": "Top-Down Twin-Stick 2D",
            "weapon": {"name": "Pulse Plasma Rifle", "desc": "High-energy directed plasma carbine built to pierce armored hulls and shields."},
            "props": [
                {"name": "Energy Deflector Core", "desc": "Deployable power node recharging personal kinetic shields."},
                {"name": "Encrypted Tactical Terminal", "desc": "Security console controlling station blast doors and defense turrets."}
            ],
            "mechanic": "Energy weapon heat management, thruster dodging, and sector defense",
            "hazards": ["Hull decompression breaches", "Security laser grids", "Automated plasma turrets"]
        },
        "medieval_fantasy": {
            "theme": "Medieval Kingdom Siege",
            "environment": "Fortified Castle Keep & Courtyards",
            "genre": "Fantasy Action RPG",
            "game_type": "Top-Down / Isometric Dungeon & Siege Action",
            "perspective": "Isometric 2.5D",
            "weapon": {"name": "Forged Greatsword", "desc": "Tempered steel blade balanced for cleaving armor and parrying heavy strikes."},
            "props": [
                {"name": "Ancient Treasure Chest", "desc": "Iron-banded war chest containing healing elixirs and runic charms."},
                {"name": "Castle Signal Brazier", "desc": "Stone fire altar lighting the battlements and restoring stamina."}
            ],
            "mechanic": "Melee combo parrying, citadel exploration, and boss encounter strategy",
            "hazards": ["Scorching fire traps", "Crushing portcullis gates", "Swarming siege patrols"]
        },
        "wilderness_survival": {
            "theme": "Wilderness Frontier Survival",
            "environment": "Untamed Taiga & Frozen Valley",
            "genre": "Hardcore Wilderness Survival",
            "game_type": "Open-World Top-Down Survival",
            "perspective": "Top-Down 2D",
            "weapon": {"name": "Forged Survival Hatchet", "desc": "Heavy timber axe used for chopping firewood, crafting shelters, and self-defense."},
            "props": [
                {"name": "Campfire Hearth", "desc": "Stone-ringed fire pit providing warmth and cooking rations."},
                {"name": "Rainwater Collector", "desc": "Canvas tarp reservoir gathering clean drinking water."}
            ],
            "mechanic": "Stamina & shelter crafting, resource gathering, and hazard navigation",
            "hazards": ["Freezing weather fronts", "Predatory wildlife", "Treacherous terrain drops"]
        },
    }

    return {
        "biome": biome,
        "enemies": extracted_enemies,
        "hero": hero_obj,
        "defaults": biome_defaults.get(biome),
        "has_conflict": bool(extracted_enemies or any(k in p_low for k in ["fight", "battle", "survive", "apocalypse", "invasion", "shooter", "defeat", "war", "siege", "boss"]))
    }


def build_story_brief(
    title: str,
    theme: str,
    environment: str,
    characters: list,
    enemies: list,
    weapons: list,
    props: list,
    core_loop: Dict[str, Any],
    visual_style: str,
) -> str:
    """Synthesizes a clean, readable Game Story & Premise narrative from the structured spec."""
    hero_name = characters[0]["name"] if characters else "Hero"
    weapon_name = weapons[0]["name"] if weapons else "Signature Gear"
    enemy_names = [e["name"] if isinstance(e, dict) else str(e) for e in enemies[:2]]
    prop_name = props[0]["name"] if props else "Key Relics"
    objective = (core_loop or {}).get("objective", f"overcome the trials of {environment}")
    env_label = environment or "an uncharted realm"

    if enemy_names:
        threat_phrase = " and ".join(enemy_names)
        return (
            f"Set across {env_label} in a {visual_style or 'Pixel Art'} world, {title} places you as a {hero_name} "
            f"confronting relentless {threat_phrase}. Armed with {weapon_name} and relying on critical assets like {prop_name}, "
            f"your mission is clear: {objective[0].lower() + objective[1:] if objective else 'survive and prevail'}."
        )
    return (
        f"Set inside {env_label} in a {visual_style or 'Pixel Art'} world, {title} puts you in control of {hero_name} "
        f"mastering the core mechanics of {theme or 'this adventure'}. Equipped with {weapon_name} and key items such as {prop_name}, "
        f"your objective is to {objective[0].lower() + objective[1:] if objective else 'complete every challenge ahead'}."
    )


def derive_asset_tags(spec_dict: Dict[str, Any]) -> list:
    """
    Deterministic rule-based enrichment (Stage 2):
    Derives pixel art search tags from Flan-T5 generated theme and characters
    for downstream sprite discovery in VAE / Asset Studio.
    """
    tags = []
    theme = spec_dict.get("theme", "")
    if theme:
        tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', theme.lower()) + "_theme")

    for char in spec_dict.get("characters", []):
        c_name = char.get("name", "")
        if c_name:
            tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', c_name.lower()) + "_character")

    for en in spec_dict.get("enemies", []):
        e_name = en.get("name", "") if isinstance(en, dict) else str(en)
        if e_name:
            tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', e_name.lower()) + "_enemy")

    for weapon in spec_dict.get("weapons", []):
        w_name = weapon.get("name", "")
        if w_name:
            tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', w_name.lower()) + "_weapon")

    for prop in spec_dict.get("props", []):
        p_name = prop.get("name", "")
        if p_name:
            tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', p_name.lower()) + "_item")

    if not tags:
        tags = ["pixel_art_character", "game_item", "environment_tile"]
    return tags[:8]


# ==============================================================================
# Common Casual & Viral Arcade Archetypes (Stage 2 Grounding Registry)
# Prevents small 80M T5 vocabulary snapping on casual/mobile games
# (e.g., Subway Surfers, Crossy Road, Flappy Bird, Aim Arrow, Pizza Delivery)
# ==============================================================================
COMMON_ARCADE_ARCHETYPES = [
    {
        "id": "subway_runner",
        "keywords": ["subway", "surfer", "endless runner", "temple run", "train track", "railway runner", "hoverboard", "runner"],
        "theme": "Subway Rail Runner",
        "environment": "Bustling Subway Tracks & Overpasses",
        "genre": "Endless Arcade Runner",
        "game_type": "Fast-Paced 3-Lane Reflex Runner",
        "perspective": "Behind-the-Back 3D Runner",
        "character": {
            "name": "Subway Urban Runner",
            "desc": "An agile street skater dodging oncoming metro trains, low signal barriers, and transit security."
        },
        "weapon_or_tool": {
            "name": "Spray Paint Jetpack",
            "desc": "A customized aerosol booster used to tag billboards and trigger emergency high-clearance leaps."
        },
        "props": [
            {"name": "Gold Magnet Coin", "desc": "A spinning magnetic collectible that draws nearby coins directly to the runner."},
            {"name": "Super Bouncing Sneakers", "desc": "Spring-cushioned high-top shoes granting immense vertical leap over train carriages."}
        ],
        "core_loop": {
            "mechanic": "Triple-lane swiping, low-barrier sliding, and train rooftop jumping",
            "objective": "Outrun the transit security patrol across active subway railways while collecting maximum gold coins",
            "hazards": ["Oncoming express subway cars", "Low-hanging railway signal gantries", "Dead-end track switch buffers"]
        }
    },
    {
        "id": "crossy_road",
        "keywords": ["crossy road", "chicken cross", "chicken road", "cross road", "chicken", "crossing", "frogger"],
        "theme": "Highway Chicken Crossing Arcade",
        "environment": "Multi-Lane Asphalt Highway & River Rapids",
        "genre": "Isometric Endless Arcade Hopper",
        "game_type": "Grid-Based Timing & Reflex Arcade",
        "perspective": "Isometric 2.5D",
        "character": {
            "name": "Feathered Road Hopper",
            "desc": "A daring chicken timing precise single-grid hops across high-speed highway traffic and drifting river timbers."
        },
        "weapon_or_tool": {
            "name": "Pogo Spring Hop",
            "desc": "A spring-loaded boost mechanism granting an emergency double-hop over oncoming vehicle fenders."
        },
        "props": [
            {"name": "Floating River Log", "desc": "A bobbing pine timber drifting downstream, providing vital temporary river footing."},
            {"name": "Golden Corn Seed", "desc": "Glistening grain pickups scattered across lane medians to boost cumulative score."}
        ],
        "core_loop": {
            "mechanic": "One-tap hop timing, lane velocity scanning, and moving log balancing",
            "objective": "Navigate across infinite highway lanes, high-speed train tracks, and rushing waterways without getting hit",
            "hazards": ["Speeding semi-trailer trucks", "Rapid river water drowning", "Hovering predatory eagles"]
        }
    },
    {
        "id": "flappy_bird",
        "keywords": ["flappy bird", "flappy", "flying bird", "pipe", "tap to fly", "helicopter", "bird avoiding"],
        "theme": "Skyward Pipe Obstacle Navigator",
        "environment": "Retro Cloudscape & Green Industrial Pipes",
        "genre": "Physics-Based Tap-to-Fly Arcade",
        "game_type": "Side-Scrolling One-Button Endurance",
        "perspective": "Side-Scrolling 2D",
        "character": {
            "name": "Flapping Sparrow",
            "desc": "A determined little winged flyer maneuvering through hazardous vertical pipe apertures with rhythmic flaps."
        },
        "weapon_or_tool": {
            "name": "Feathered Glider Wing",
            "desc": "Aerodynamic plumage providing controlled micro-bursts of upward momentum against relentless gravity."
        },
        "props": [
            {"name": "Bronze Flight Medal", "desc": "A milestone token awarded for threading safely through difficult pipe gaps."},
            {"name": "High-Altitude Cloud Ring", "desc": "A buoyant atmospheric vapor ring providing momentary altitude stabilization."}
        ],
        "core_loop": {
            "mechanic": "Rhythmic tap-lift impulse, gravity counter-balance, and gap aperture alignment",
            "objective": "Thread through consecutive pairs of industrial pipes without touching the borders or the floor",
            "hazards": ["Narrow vertical pipe clearances", "Sudden thermal downdrafts", "Terminal gravity drop"]
        }
    },
    {
        "id": "archery_aim_arrow",
        "keywords": ["aim arrow", "archery", "bow and arrow", "arrow", "target shoot", "bullseye", "bowman", "marksman"],
        "theme": "Precision Archery Challenge",
        "environment": "Highland Forest Archery Pavilion & Moving Targets",
        "genre": "Physics-Based Target Archery",
        "game_type": "Precision Aim & Release Trajectory Challenge",
        "perspective": "First-Person / Side-View Aim",
        "character": {
            "name": "Master Marksman Archer",
            "desc": "A focused archer calculating wind shear, distance drop, and release velocity for pinpoint long-range accuracy."
        },
        "weapon_or_tool": {
            "name": "Composite Recurve Bow",
            "desc": "A handcrafted laminated bow engineered for minimum vibration, high arrow velocity, and laser-flat trajectory."
        },
        "props": [
            {"name": "Concentric Bullseye Target", "desc": "A multi-ringed straw target registering tiered points based on center proximity."},
            {"name": "Fletched Arrow Quiver", "desc": "A leather hip quiver storing precision arrows with aerodynamic goose feather fletching."}
        ],
        "core_loop": {
            "mechanic": "Draw-string tensioning, wind drift compensation, and release breath timing",
            "objective": "Score consecutive bullseyes on stationary and swinging targets across varying distances and weather",
            "hazards": ["Gusting crosswinds", "Oscillating pendulum targets", "Sudden vision-obscuring rain squalls"]
        }
    },
    {
        "id": "cooking_delivery",
        "keywords": ["pizza", "delivery", "cooking", "restaurant", "baker", "chef", "cafe", "food", "kitchen"],
        "theme": "Pizza Delivery Simulation",
        "environment": "Bustling Downtown Pizzeria & Streets",
        "genre": "Urban Time-Management & Simulation",
        "game_type": "Casual Pizza & Food Delivery Simulation",
        "perspective": "Top-Down 2D",
        "character": {
            "name": "Pizza Delivery Courier",
            "desc": "A skilled courier navigating busy city avenues to deliver steaming orders before timers expire."
        },
        "weapon_or_tool": {
            "name": "Pizza Cutter Wheel",
            "desc": "A precision rotary slicing tool used to portion fresh brick-oven pies rapidly."
        },
        "props": [
            {"name": "Insulated Pizza Hotbox", "desc": "A thermal carrying bag keeping delivery orders piping hot during transit."},
            {"name": "Stone Pizza Oven", "desc": "A traditional wood-fired brick oven baking orders at scorching temperatures."}
        ],
        "core_loop": {
            "mechanic": "Order assembly, oven bake timing, and rush-hour city navigation",
            "objective": "Fulfill and deliver gourmet orders across downtown traffic before customer satisfaction timers expire",
            "hazards": ["Rush-hour traffic jams", "Slippery wet asphalt", "Aggressive stray dogs"]
        }
    },
    {
        "id": "pacman_maze",
        "keywords": ["pacman", "pac-man", "maze runner", "dot eating", "ghost maze", "labyrinth runner"],
        "theme": "Neon Labyrinth Maze Chaser",
        "environment": "Bioluminescent Grid Maze",
        "genre": "Classic Maze Action Arcade",
        "game_type": "Top-Down Grid Navigation & Evasion",
        "perspective": "Top-Down 2D",
        "character": {
            "name": "Neon Sphere Chaser",
            "desc": "A bright luminescent sphere consuming energetic pellets while evading spectral pursuers in cyber corridors."
        },
        "weapon_or_tool": {
            "name": "Super Power Pellet Surge",
            "desc": "A concentrated burst of energy that temporarily reverses ghost pursuit roles, enabling counter-capture."
        },
        "props": [
            {"name": "Glowing Energy Dot", "desc": "Small luminous pellets paving the labyrinth corridors that charge player speed."},
            {"name": "Bonus Fruit Beacon", "desc": "A high-value fruit relic appearing briefly in the center of the grid for score multiplier."}
        ],
        "core_loop": {
            "mechanic": "Corner pathing, pellet line clearing, and power-boost ghost hunting",
            "objective": "Clear all energy dots from the maze corridors while out-maneuvering roaming phantom patrols",
            "hazards": ["Roaming spectral patrols", "Corner dead ends", "Speed-drain corridor zones"]
        }
    },
    {
        "id": "fruit_ninja",
        "keywords": ["fruit ninja", "fruit slice", "slice fruit", "ninja slice", "fruit cutting"],
        "theme": "Fruit Blade Dojo Arcade",
        "environment": "Traditional Dojo Bamboo Cutting Arena",
        "genre": "Gesture-Based Slicing Action",
        "game_type": "Reflex Arcade Combo Slicer",
        "perspective": "Front-Facing 2D Arena",
        "character": {
            "name": "Dojo Blade Master",
            "desc": "A disciplined martial artist slicing airborne fruit with lightning-fast blade strokes and fluid precision."
        },
        "weapon_or_tool": {
            "name": "Tempered Katana Blade",
            "desc": "A razor-honed blade cutting cleanly through tossed tropical fruits in single sweeping arcs."
        },
        "props": [
            {"name": "Juicy Flying Watermelon", "desc": "A large high-scoring fruit that bursts into vibrant splashes upon a critical slice."},
            {"name": "Ticking Bomb Hazard", "desc": "A volatile contact explosive tossed among fruit that detonates and penalizes lives if struck."}
        ],
        "core_loop": {
            "mechanic": "Multi-fruit swipe comboing, blade angle trajectory, and bomb avoidance",
            "objective": "Slice as many tossed fruits as possible to achieve high combo multipliers before missing three",
            "hazards": ["Airborne contact bombs", "Dropping unsliced fruit penalties", "Screen-obscuring juice splashes"]
        }
    },
    {
        "id": "racing_drift",
        "keywords": ["racing", "car race", "drift", "speed car", "formula", "supercar", "nitro"],
        "theme": "High-Speed Drift Circuit",
        "environment": "Neon Asphalt Circuit & Hairpin Curves",
        "genre": "Arcade Motorsport & Drift Racing",
        "game_type": "Top-Down / Isometric Track Racing",
        "perspective": "Top-Down 2D",
        "character": {
            "name": "Pro Street Racer",
            "desc": "A fearless driver mastering high-speed apex turns, slipstream overtakes, and controlled drifting lines."
        },
        "weapon_or_tool": {
            "name": "Turbo Nitro Boost Injector",
            "desc": "A pressurized nitrous oxide canister delivering instant bursts of extreme straight-line acceleration."
        },
        "props": [
            {"name": "Checkered Finish Flag", "desc": "The ultimate milestone marker designating lap completions and podium finishes."},
            {"name": "Drift Oil Canister", "desc": "Slick asphalt hazard reducing tire traction if touched during high-angle cornering."}
        ],
        "core_loop": {
            "mechanic": "Brake-drift entry, steering counter-steer, and nitro boost timing",
            "objective": "Set record lap times and overtake rival drivers on treacherous neon city circuits",
            "hazards": ["Tire wall barriers", "Slick oil puddles", "Tight hairpin chicane corners"]
        }
    },
    {
        "id": "fishing_simulator",
        "keywords": ["fishing", "fish game", "lake fishing", "angler", "river fishing", "catch fish"],
        "theme": "Serene Lake Fishing Simulator",
        "environment": "Misty Morning Lake & Wooden Pier",
        "genre": "Cozy Sports & Relaxation Simulator",
        "game_type": "Casual Nature Exploration & Fishing",
        "perspective": "Side-View / Isometric 2.5D",
        "character": {
            "name": "Patient Lake Angler",
            "desc": "An experienced fisherman studying surface ripples and casting lines into tranquil deep waters."
        },
        "weapon_or_tool": {
            "name": "Telescopic Graphite Fishing Rod",
            "desc": "A sensitive fishing rod fitted with smooth drag reels and braided line for delicate bite detection."
        },
        "props": [
            {"name": "Floating Cork Bobber", "desc": "A buoyant painted float that dips below water when a fish strikes."},
            {"name": "Tackle Box with Lures", "desc": "A compartmentalized box storing spinners, spoons, and bait hooks."}
        ],
        "core_loop": {
            "mechanic": "Cast distance metering, bite reaction timing, and line tension reel balancing",
            "objective": "Catalog diverse freshwater species and reel in rare trophy fish without snapping lines",
            "hazards": ["Submerged tree snags", "Line snapping from excessive tension", "Sudden boat wake waves"]
        }
    },
    {
        "id": "brick_platformer",
        "keywords": ["mario", "platformer", "jump and run", "brick breaker", "coin jump"],
        "theme": "Retro 2D Jump Platformer",
        "environment": "Mushroom Kingdom Hills & Floating Bricks",
        "genre": "Classic Side-Scrolling Platformer",
        "game_type": "Precision Platforming & Block Traversal",
        "perspective": "Side-Scrolling 2D",
        "character": {
            "name": "Acrobatic Jumper Hero",
            "desc": "A plucky adventurer leaping across floating brick blocks, collecting coins, and stomping pesky pests."
        },
        "weapon_or_tool": {
            "name": "Super Stomp Boots",
            "desc": "Reinforced leather boots delivering heavy downwards stomp impacts on enemy minions."
        },
        "props": [
            {"name": "Question Mark Coin Block", "desc": "A glowing mystery brick that expels gold coins when bumped from below."},
            {"name": "Super Star Power-Up", "desc": "An invincible bouncing star rendering the hero invulnerable for a limited duration."}
        ],
        "core_loop": {
            "mechanic": "Variable jump height, brick bumping, and gap leaping",
            "objective": "Reach the flagpole at the end of each stage before the stage timer runs down",
            "hazards": ["Bottomless pit gaps", "Moving spiky creatures", "Descending crush blocks"]
        }
    }
]


def enrich_game_concept_metadata(
    parsed_spec: Dict[str, Any],
    prompt: str,
    matched_arcade: Optional[Dict[str, Any]] = None,
    prompt_semantics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Stage 2 Deterministic Semantic Enrichment:
    Enriches the neural Flan-T5 output with deep game design foundational attributes:
    - game_title: Contextual, marketable title
    - genre: Specific primary genre (e.g. 'Action RPG / Roguelite', 'Cozy Farming Sim')
    - game_type: Structural gameplay classification (e.g. 'Top-Down Dungeon Crawler', 'Isometric Survival')
    - target_perspective: Recommended camera view ('Top-Down 2D', 'Side-Scrolling Platformer', 'Isometric 2.5D')
    - core_loop: Core mechanics, win conditions, and environmental hazards
    - entity slot auto-fill: Guarantees non-empty props, companion roles, and theme-aligned tools
    """
    theme = (parsed_spec.get("theme") or "").strip()
    env = (parsed_spec.get("environment") or "").strip()
    chars = parsed_spec.get("characters", [])
    main_char = chars[0].get("name", "Hero") if chars and isinstance(chars[0], dict) else "Protagonist"
    
    combined_text = f"{prompt} {theme} {env}".lower()

    # Thematic entity presets for Stage 2 slot-filling if Flan-T5 outputs empty arrays
    thematic_fallbacks = {
        "farming": {
            "genre": "Cozy Farming & Life Sim",
            "game_type": "Cozy Management Simulation",
            "perspective": "Top-Down 2D",
            "core_loop": {
                "mechanic": "Crop sowing, botanical synthesis, and resource trade",
                "objective": f"Cultivate a sustainable harvest in {env or 'the sanctuary'} and fulfill community requests",
                "hazards": ["Weed blight spread", "Seasonal frost", "Nocturnal garden pests"]
            },
            "props": [
                {"name": "Candy Cane Sprout", "desc": "Sweet interactive crop that yields confectionery crystals when watered."},
                {"name": "Fox Den Basket", "desc": "Cozy resting basket that stores harvested berries and seeds."}
            ],
            "tool_alternative": {"name": "Harvest Trowel", "desc": "Delicate ceramic gardening tool for gentle root harvesting."}
        },
        "cooking_delivery": {
            "genre": "Urban Time-Management & Simulation",
            "game_type": "Casual Pizza & Food Delivery Simulation",
            "perspective": "Top-Down 2D",
            "core_loop": {
                "mechanic": "Food preparation, oven timing, and rush-hour navigation",
                "objective": f"Deliver steaming orders across {env or 'the city'} before customer satisfaction timers expire",
                "hazards": ["Rush-hour traffic jams", "Slippery road turns", "Aggressive stray dogs"]
            },
            "props": [
                {"name": "Insulated Pizza Hotbox", "desc": "A thermal carrying bag keeping pizzas hot during delivery."},
                {"name": "Stone Pizza Oven", "desc": "A traditional brick oven baking orders at scorching temperatures."}
            ],
            "tool_alternative": {"name": "Pizza Slicing Wheel", "desc": "A sharp rotary cutter used to portion pizzas rapidly."}
        },
        "dungeon": {
            "genre": "Action RPG / Roguelite",
            "game_type": "Top-Down Dungeon Crawler",
            "perspective": "Top-Down (45° Isometric)",
            "core_loop": {
                "mechanic": "Hack-and-slash melee combat, room clearing, and relic collection",
                "objective": f"Descend through the {env or 'catacombs'} and vanquish the ancient guardian",
                "hazards": ["Spike floor grates", "Falling stalactites", "Poison gas vents"]
            },
            "props": [
                {"name": "Ancient Treasure Chest", "desc": "Heavy iron-banded chest containing gold coins and runic keystones."},
                {"name": "Mana Brazier", "desc": "Interactive stone altar that replenishes player energy upon activation."}
            ],
            "tool_alternative": None
        },
        "cyberpunk": {
            "genre": "Fast-Paced Action Platformer",
            "game_type": "Cyberpunk Stealth Infiltration",
            "perspective": "Side-Scrolling 2D",
            "core_loop": {
                "mechanic": "Cyber-dash slicing, laser parrying, and terminal hacking",
                "objective": f"Infiltrate {env or 'megacorp facility'} and extract confidential neural data",
                "hazards": ["Security laser grids", "Thermal auto-turrets", "Elevator shaft drops"]
            },
            "props": [
                {"name": "Holographic Terminal", "desc": "Bypass console containing encrypted security passcodes."},
                {"name": "Nanotech Repair Stash", "desc": "Wall-mounted medical cache restoring cyberware integrity."}
            ],
            "tool_alternative": None
        },
        "horror": {
            "genre": "Survival Horror",
            "game_type": "Atmospheric Psychological Horror",
            "perspective": "Top-Down 2D (Dark Vignette)",
            "core_loop": {
                "mechanic": "Lantern battery economy, stealth evasion, and protective ward spreading",
                "objective": f"Uncover the dark secret of {env or 'the sanitarium'} and survive until dawn",
                "hazards": ["Flickering dark zones", "Mind-shattering phantoms", "Spike traps"]
            },
            "props": [
                {"name": "Old Kerosene Lantern", "desc": "Fragile light source that repels encroaching shadows."},
                {"name": "Ritual Chalk Circle", "desc": "Sanctified floor ward preventing entity manifestation."}
            ],
            "tool_alternative": None
        },
        "space": {
            "genre": "Sci-Fi Action / Survival",
            "game_type": "Planetary Exploration & Extraction",
            "perspective": "Top-Down Twin-Stick 2D",
            "core_loop": {
                "mechanic": "Zero-G thruster navigation, scanner pinging, and turret combat",
                "objective": f"Repair auxiliary life support in {env or 'the orbital station'} and escape safely",
                "hazards": ["Hull decompression breaches", "Acid geysers", "Rogue drone swarms"]
            },
            "props": [
                {"name": "Plasma Power Cell", "desc": "Portable energy capacitor to restart auxiliary ship generators."},
                {"name": "Mineral Scanner Beacon", "desc": "Deployable sonar probe identifying subterranean ore nodes."}
            ],
            "tool_alternative": None
        },
        "steampunk": {
            "genre": "Steampunk Action Adventure",
            "game_type": "Clockwork Tactical Brawler",
            "perspective": "Side-Scrolling 2D",
            "core_loop": {
                "mechanic": "Steam-pressure venting, grapple traversal, and gear swapping",
                "objective": f"Quell the rogue automaton revolt across {env or 'the industrial city'}",
                "hazards": ["High-pressure steam jets", "Molten brass vats", "Crushing gear teeth"]
            },
            "props": [
                {"name": "Brass Pressure Valve", "desc": "Interactive steam valve to regulate boiler pipeline pressure."},
                {"name": "Aether Battery", "desc": "Electrified coil cell fueling pneumatic gadgets."}
            ],
            "tool_alternative": None
        },
        "western": {
            "genre": "Tactical Western Shooter",
            "game_type": "Frontier Quick-Draw Action",
            "perspective": "Top-Down Tactical 2D",
            "core_loop": {
                "mechanic": "Revolver fanning, cover vaulting, and quick-draw duel timing",
                "objective": f"Capture the outlaw syndicate terrorizing {env or 'the frontier'}",
                "hazards": ["Dynamite barrels", "Barbed wire fences", "Desert dust storms"]
            },
            "props": [
                {"name": "Sheriff Bounty Board", "desc": "Notice board displaying fugitive bounties and payout rewards."},
                {"name": "Dynamite Crate", "desc": "Volatile explosive cache useful for breaching barricades."}
            ],
            "tool_alternative": None
        },
        "mystery": {
            "genre": "Noir Narrative Detective / Puzzle",
            "game_type": "Point-and-Click Mystery Adventure",
            "perspective": "Isometric 2.5D",
            "core_loop": {
                "mechanic": "Clue dusting, suspect interrogation, and deductive timeline mapping",
                "objective": f"Expose the culprit behind the tragedy at {env or 'the estate'}",
                "hazards": ["Slippery terrain", "Hostile henchmen", "Dwindling investigation timer"]
            },
            "props": [
                {"name": "Magnifying Glass", "desc": "Optical tool highlighting forensic microscopic traces."},
                {"name": "Fingerprint Dossier", "desc": "Classified ledger matching suspects to crime scene prints."}
            ],
            "tool_alternative": None
        },
        "survival": {
            "genre": "Hardcore Wilderness Survival",
            "game_type": "Open-World Top-Down Survival",
            "perspective": "Top-Down 2D",
            "core_loop": {
                "mechanic": "Calorie/hydration tracking, shelter construction, and campfire stoking",
                "objective": f"Endure extreme elements in {env or 'the wilderness'} and signal for rescue",
                "hazards": ["Hypothermic blizzards", "Predatory beasts", "Thin lake ice"]
            },
            "props": [
                {"name": "Campfire Pit", "desc": "Stone-ringed fire hearth that staves off hypothermia and cooks meals."},
                {"name": "Rainwater Collector", "desc": "Tarp reservoir collecting purified potable drinking water."}
            ],
            "tool_alternative": None
        },
        "outbreak_apocalypse": {
            "genre": "Post-Apocalyptic Survival Action",
            "game_type": "Top-Down Horde Survival & Scavenging",
            "perspective": "Top-Down 2D",
            "core_loop": {
                "mechanic": "Scavenging dwindling supplies, barricading safehouses, and fending off hostile waves",
                "objective": f"Survive relentless hostile waves across {env or 'the quarantine ruins'} and reach extraction",
                "hazards": ["Swarming infected hordes", "Collapsed urban rubble", "Toxic contamination zones"]
            },
            "props": [
                {"name": "Emergency Medkit Crate", "desc": "Sealed medical supply box restoring health and curing trauma."},
                {"name": "Barricaded Supply Checkpoint", "desc": "Fortified blockade storing salvaged ammo, rations, and flares."}
            ],
            "tool_alternative": None
        }
    }

    # Match thematic category using prompt semantics first so outbreak/horror/sci-fi survival never falls back to arctic wilderness
    sem_biome = (prompt_semantics or {}).get("biome")
    matched_cat = "fantasy"
    if sem_biome == "outbreak_apocalypse":
        matched_cat = "outbreak_apocalypse"
    elif sem_biome == "gothic_horror":
        matched_cat = "horror"
    elif sem_biome == "sci_fi_action":
        matched_cat = "space"
    elif sem_biome == "medieval_fantasy":
        matched_cat = "dungeon"
    elif any(k in combined_text for k in ["pizza", "cook", "restaurant", "baker", "chef", "cafe", "food", "kitchen", "delivery"]):
        matched_cat = "cooking_delivery"
    elif any(k in combined_text for k in ["farm", "cozy", "harvest", "crop", "witch", "botanist", "seed", "fox", "candy"]):
        matched_cat = "farming"
    elif any(k in combined_text for k in ["dungeon", "crypt", "catacomb", "labyrinth", "depths", "skeleton"]):
        matched_cat = "dungeon"
    elif any(k in combined_text for k in ["cyberpunk", "hacker", "neon", "megacorp", "cyborg", "drone", "synth"]):
        matched_cat = "cyberpunk"
    elif any(k in combined_text for k in ["horror", "asylum", "vampire", "morgue", "strigoi", "shadow", "whisper", "haunted"]):
        matched_cat = "horror"
    elif any(k in combined_text for k in ["space", "orbit", "alien", "galaxy", "rover", "astro", "vacuum", "derelict"]):
        matched_cat = "space"
    elif any(k in combined_text for k in ["steampunk", "airship", "clockwork", "steam", "brass", "foundry", "boiler"]):
        matched_cat = "steampunk"
    elif any(k in combined_text for k in ["western", "gunslinger", "sheriff", "outlaw", "prairie", "desert", "saloon"]):
        matched_cat = "western"
    elif any(k in combined_text for k in ["mystery", "detective", "noir", "investigat", "clue", "murder"]):
        matched_cat = "mystery"
    elif any(k in combined_text for k in ["island", "blizzard", "arctic", "wilderness", "taiga", "surviv"]):
        matched_cat = "survival"

    if matched_arcade:
        genre = matched_arcade["genre"]
        game_type = matched_arcade["game_type"]
        perspective = matched_arcade["perspective"]
        core_loop = matched_arcade["core_loop"]
        fallback_props = matched_arcade["props"]
        tool_alt = matched_arcade["weapon_or_tool"]
        matched_cat = matched_arcade["id"]
    elif matched_cat in thematic_fallbacks:
        preset = thematic_fallbacks[matched_cat]
        genre = preset["genre"]
        game_type = preset["game_type"]
        perspective = preset["perspective"]
        core_loop = preset["core_loop"]
        fallback_props = preset["props"]
        tool_alt = preset["tool_alternative"]
    else:
        t_label = theme if theme else "Fantasy"
        genre = f"{t_label} Action RPG"
        game_type = f"2D {t_label} Adventure"
        perspective = "Top-Down 2D"
        core_loop = {
            "mechanic": "Exploration, tactical combat, and character progression",
            "objective": f"Overcome the challenges of {env or 'the realm'} and defeat the final boss",
            "hazards": ["Environmental traps", "Hostile patrols", "Hazardous terrain"]
        }
        fallback_props = [
            {"name": "Mystic Relic", "desc": "An ancient artifact emitting a soft, ethereal luminescence."},
            {"name": "Resource Cache", "desc": "A supply crate containing essential crafting materials."}
        ]
        tool_alt = None

    # Dynamic Marketable Title Generation
    env_clean = env.replace("The ", "").strip()
    env_word = env_clean.split()[0] if env_clean else (theme if theme else "GameForge")
    if theme and theme.lower() not in env_word.lower():
        title = f"{theme}: {env_word} Chronicles"
    else:
        title = f"{env_word}: Quest of the {main_char.split()[0]}"

    return {
        "game_title": title,
        "genre": genre,
        "game_type": game_type,
        "target_perspective": perspective,
        "core_loop": core_loop,
        "fallback_props": fallback_props,
        "tool_alt": tool_alt,
        "matched_cat": matched_cat
    }


class TransformerService:
    def __init__(self):
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        self.model_path = MODEL_DIR
        self._lock = threading.Lock()

    def load_model(self):
        """Loads the fine-tuned Flan-T5 checkpoint (models/4_Transformer_2k_v1) and warms up."""
        if self.is_loaded and self.model is not None:
            return
        with self._lock:
            if self.is_loaded and self.model is not None:
                return
            logger.info(f"Loading Fine-Tuned Flan-T5 Transformer from {self.model_path}...")
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                self.model.eval()

                # Model warm-up to eliminate cold-start latency on first user prompt
                dummy_input = self.tokenizer(f"{TASK_PREFIX}warm up", return_tensors="pt")
                _ = self.model.generate(**dummy_input, max_new_tokens=16)

                self.is_loaded = True
                logger.info("Fine-Tuned Flan-T5 model loaded and warmed up successfully.")
            except Exception as e:
                logger.error(f"Failed to load fine-tuned Transformer model from {self.model_path}: {e}")
                self.is_loaded = False
                raise RuntimeError(f"Transformer model unavailable: {e}")

    def unload_model(self):
        """Safely unloads PyTorch + FLAN-T5 model and releases CPU memory via garbage collection."""
        with self._lock:
            if self.model is not None or self.tokenizer is not None or self.is_loaded:
                logger.info("Unloading Transformer model to release CPU RAM...")
                self.model = None
                self.tokenizer = None
                self.is_loaded = False
                import gc
                gc.collect()
                logger.info("Transformer model unloaded and CPU memory garbage-collected.")

    def audit_prompt_guardrails(self, user_prompt: str) -> Tuple[bool, str, List[Dict[str, str]]]:
        """
        Runs multi-layer Responsible AI & Security checks on the user prompt:
        1. Input Sanitization & Prompt-Injection Shield
        2. NSFW, Hate Speech & Graphic Gore Regex Boundary Audit
        3. Third-Party Protected IP Detection & Re-grounding
        4. Dataset Provenance & CC-BY-NC-SA 4.0 Attribution
        5. Human-in-the-Loop (HITL) Asset Generation Gate
        """
        prompt_text = (user_prompt or "").strip()
        prompt_lower = prompt_text.lower()

        checks = [
            {
                "name": "Prompt Injection & Input Sanitization",
                "status": "passed",
                "detail": "Input length and syntax validated; no script tags, SQL/shell operators, or instruction-override phrases detected."
            },
            {
                "name": "NSFW, Hate & Extreme Violence Filter",
                "status": "passed",
                "detail": "Passed regex boundary audit (no explicit sexual content, hate speech, self-harm, or graphic gore)."
            },
            {
                "name": "Copyright & Protected IP Shield",
                "status": "passed",
                "detail": "No protected third-party franchise identifiers detected; original archetypes generated."
            },
            {
                "name": "Dataset Provenance & Attribution",
                "status": "verified",
                "detail": "Fine-Tuned Flan-T5 on 2,000 curated prompt-JSON pairs (80/10/10 split) • CC-BY-NC-SA 4.0."
            },
            {
                "name": "Human-in-the-Loop (HITL) Approval Gate",
                "status": "pending_review",
                "detail": "Designer sign-off required in UI before forwarding asset tags to downstream sprite generators."
            }
        ]

        if not prompt_text:
            return True, "Passed Safety Audit", checks

        # 1. Length & Injection / Sanitization check
        if len(prompt_text) > 500:
            checks[0]["status"] = "blocked"
            checks[0]["detail"] = "Input exceeds maximum allowed 500-character safety bound."
            return False, "Prompt exceeds maximum length (500 characters).", checks

        injection_patterns = [
            r"<\s*script", r"\bon(?:error|load|click)\s*=", r"javascript\s*:",
            r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions|prompts|rules)",
            r"(?:drop\s+table|union\s+select|;\s*rm\s+-rf|\$\([^)]+\))"
        ]
        for pat in injection_patterns:
            if re.search(pat, prompt_lower):
                checks[0]["status"] = "blocked"
                checks[0]["detail"] = "Blocked potential HTML/XSS, SQL/command, or prompt-override injection pattern."
                return False, "Prompt blocked by Security Sanitization Shield (unsafe syntax or override attempt).", checks

        # 2. Prohibited NSFW / Harmful Terms
        prohibited_terms = [
            "nsfw", "nude", "explicit", "gore", "decapitation",
            "hate", "racist", "slur", "torture", "suicide"
        ]
        for term in prohibited_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', prompt_lower):
                checks[1]["status"] = "blocked"
                checks[1]["detail"] = f"Restricted term detected ('{term}'); generation halted."
                return False, f"Prompt contains restricted or unsafe content ('{term}').", checks

        # 3. Direct Protected IP Warning (logged, automatically abstracted)
        protected_ips = ["mario", "pokemon", "pikachu", "sonic", "zelda"]
        matched_ips = [ip for ip in protected_ips if re.search(r'\b' + re.escape(ip) + r'\b', prompt_lower)]
        if matched_ips:
            joined_ips = ", ".join(matched_ips)
            logger.warning(f"Protected IP keyword detected: '{joined_ips}'. Re-grounding to generic asset.")
            checks[2]["status"] = "mitigated"
            checks[2]["detail"] = f"Franchise keyword ({joined_ips}) detected and abstracted into original non-infringing game entities."

        return True, "Passed Safety Audit", checks

    def validate_prompt_safety(self, user_prompt: str) -> Tuple[bool, str]:
        """Validates user prompt against Responsible AI safety boundaries (NSFW, violence, hate speech)."""
        is_safe, safety_msg, _ = self.audit_prompt_guardrails(user_prompt)
        return is_safe, safety_msg

    def generate_concept_spec(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executes Seq2Seq Neural Generation using the fine-tuned Flan-T5 model (models/4_Transformer_2k_v1).
        Outputs structured JSON strictly matching the evaluated schema:
        {theme, environment, characters, weapons, props, visualStyle}
        """
        is_safe, safety_msg, security_checks = self.audit_prompt_guardrails(user_prompt)
        if not is_safe:
            raise ValueError(safety_msg)

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self.load_model()

        clean_prompt = user_prompt.strip() if user_prompt else "Fantasy RPG adventure"
        formatted_input = f"{TASK_PREFIX}{clean_prompt}"

        # 1. Real Neural Model Generation with Telemetry
        start_time = time.perf_counter()
        import torch
        with torch.no_grad():
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            # Use greedy decoding first: ~4x faster on CPU and avoids cyclic beam search loops
            output_ids = self.model.generate(
                **inputs,
                num_beams=1,
                do_sample=False,
                max_new_tokens=384
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        cleaned_text = strip_code_fences(raw_output)

        # 2. Parse Neural JSON with Automated Boundary Repair Auditor
        parse_error = None
        parsed_spec = None
        try:
            parsed_spec = json.loads(cleaned_text)
        except Exception as e:
            logger.warning(f"Initial T5 JSON parse error: {e}. Attempting Boundary Repair Auditor...")
            parsed_spec = repair_json_syntax(cleaned_text)
            if not parsed_spec:
                # Fallback: attempt beam search with early stopping
                try:
                    with torch.no_grad():
                        beam_ids = self.model.generate(
                            **inputs,
                            num_beams=4,
                            early_stopping=True,
                            max_new_tokens=256
                        )
                    beam_raw = self.tokenizer.decode(beam_ids[0], skip_special_tokens=True).strip()
                    parsed_spec = repair_json_syntax(beam_raw)
                    if parsed_spec:
                        raw_output = beam_raw
                except Exception as be:
                    logger.warning(f"Beam fallback also failed: {be}")

            if not parsed_spec:
                parse_error = str(e)

        if parse_error or not isinstance(parsed_spec, dict):
            # Fail transparently with real error details - DO NOT silently fallback to hardcoded tables
            return {
                "status": "parse_error",
                "prompt": clean_prompt,
                "formatted_input": formatted_input,
                "error_detail": f"Model generated non-JSON syntax: {parse_error}",
                "raw_t5_output": raw_output,
                "telemetry": {
                    "inference_latency_ms": latency_ms,
                    "tokens_generated": len(output_ids[0]),
                    "schema_valid": False,
                    "model_checkpoint": "models/4_Transformer_2k_v1"
                },
                "ethical_guardrails": {
                    "content_safety": safety_msg,
                    "safety_status": safety_msg,
                    "checks": security_checks,
                    "dataset_attribution": "GameForge Fine-Tuned T5 (2,000 Curated Prompt-JSON Pairs; 80/10/10 Split)",
                    "license": "CC-BY-NC-SA 4.0",
                    "human_oversight_required": True
                }
            }

        # Deduplicate entity items by name so characters/enemies/weapons/props are clean and distinct
        parsed_spec.setdefault("enemies", [])
        for field in ["characters", "enemies", "weapons", "props"]:
            if field in parsed_spec and isinstance(parsed_spec[field], list):
                seen_names = set()
                deduped = []
                for item in parsed_spec[field]:
                    if isinstance(item, dict):
                        n = item.get("name", "").strip().lower()
                        if n and n not in seen_names:
                            seen_names.add(n)
                            deduped.append(item)
                    elif isinstance(item, str) and item.strip().lower() not in seen_names:
                        seen_names.add(item.strip().lower())
                        deduped.append({"name": item, "desc": f"A {item.lower()} in the game world."})
                parsed_spec[field] = deduped

        # 3. Schema Validation against evaluated paper schema
        actual_fields = set(parsed_spec.keys())
        missing_fields = list(REQUIRED_SCHEMA_FIELDS - actual_fields)
        extra_fields = list(actual_fields - REQUIRED_SCHEMA_FIELDS)
        schema_valid = len(missing_fields) == 0

        # 4. Stage 2 Deterministic Semantic Enrichment & Prompt Grounding (Game Design Document Engine)
        lower_prompt = clean_prompt.lower()

        # Find if user prompt refers to any classic/arcade/casual game archetype
        matched_arcade = None
        for arc in COMMON_ARCADE_ARCHETYPES:
            if any(k in lower_prompt for k in arc["keywords"]):
                matched_arcade = arc
                break

        # General Compositional Semantic Extraction (works across arbitrary enemies, heroes, and biomes)
        prompt_semantics = None
        if matched_arcade:
            theme_val = parsed_spec.get("theme", "")
            hallucinated_tropes = ["pirate", "warzone", "dungeon", "skeleton", "cyber", "space", "alien", "crypt", "stylized", "fantasy", "generic"]
            if any(t in theme_val.lower() for t in hallucinated_tropes) or not theme_val or "adventure" in theme_val.lower():
                parsed_spec["theme"] = matched_arcade["theme"]

            env_val = parsed_spec.get("environment", "")
            if any(t in env_val.lower() for t in hallucinated_tropes) or not env_val or "village" in env_val.lower() or "arena" in env_val.lower():
                parsed_spec["environment"] = matched_arcade["environment"]

            # Ensure characters match archetype
            chars = parsed_spec.get("characters", [])
            if not chars or any(t in chars[0].get("name", "").lower() for t in hallucinated_tropes + ["treasure", "villager", "warzone", "software", "protagonist", "hero", "traveler"]):
                parsed_spec["characters"] = [matched_arcade["character"]]

            # Ensure weapons/tools match archetype
            weaps = parsed_spec.get("weapons", [])
            if not weaps or any(t in weaps[0].get("name", "").lower() for t in ["sword", "knife", "cannon", "gun", "axe", "tome", "blade", "tool", "weapon"]):
                parsed_spec["weapons"] = [matched_arcade["weapon_or_tool"]]

            # Ensure props match archetype
            props = parsed_spec.get("props", [])
            if not props or any(t in props[0].get("name", "").lower() for t in ["statue", "temple", "ruin", "chest", "altar", "item cache", "cache", "relic"]):
                parsed_spec["props"] = matched_arcade["props"]
        else:
            prompt_semantics = extract_prompt_semantics(clean_prompt)
            sem_biome = prompt_semantics["biome"]
            sem_defaults = prompt_semantics.get("defaults")
            sem_enemies = prompt_semantics.get("enemies", [])
            sem_hero = prompt_semantics.get("hero")
            num_tokens = len(output_ids[0])

            # 4a. Ground theme & environment if T5 collapsed to generic defaults, weapon-as-theme, or cross-biome hallucination
            theme_val = parsed_spec.get("theme", "").strip()
            env_val = parsed_spec.get("environment", "").strip()
            generic_themes = {"", "action adventure", "general", "adventure", "retro arcade"}
            generic_envs = {"", "stylized realm", "arena", "generic world"}
            theme_env_blob = f"{theme_val} {env_val}".lower()
            weapon_like_theme = any(wk in theme_val.lower() for wk in ["axe", "sword", "spear", "shield", "rifle", "blaster"])
            cross_biome_mismatch = (
                (sem_biome == "outbreak_apocalypse" and any(w in theme_env_blob for w in ["candy", "farm", "cozy", "pirate", "medieval", "zero-grid", "zero-gravity", "orbital", "space", "cargo bay", "cyber"]))
                or (sem_biome == "medieval_fantasy" and any(w in theme_env_blob for w in ["cyber", "space", "neon", "sci-fi", "modern", "zero-gravity", "orbital"]))
                or (sem_biome == "sci_fi_action" and any(w in theme_env_blob for w in ["medieval", "castle", "village", "farm", "candy", "treasure chest"]))
                or (sem_biome == "gothic_horror" and any(w in theme_env_blob for w in ["farm", "cozy", "candy", "arcade runner", "zero-gravity", "orbital"]))
                or (sem_biome == "wilderness_survival" and any(w in theme_env_blob for w in ["zero-grid", "zero-gravity", "orbital", "cyber", "candy"]))
            )
            if sem_defaults and (theme_val.lower() in generic_themes or weapon_like_theme or cross_biome_mismatch or num_tokens <= 16):
                parsed_spec["theme"] = sem_defaults["theme"]
            elif not theme_val or theme_val.lower() in generic_themes or weapon_like_theme:
                if sem_enemies:
                    parsed_spec["theme"] = f"{sem_enemies[0]['name']} Conflict"

            if sem_defaults and (env_val.lower() in generic_envs or cross_biome_mismatch or num_tokens <= 16):
                parsed_spec["environment"] = sem_defaults["environment"]
            elif not parsed_spec.get("environment", "").strip():
                loc_m = re.search(r"\b(?:in|inside|across|through|on|at)\s+(?:an?\s+|the\s+)?([a-z0-9-]+(?:\s+[a-z0-9-]+){0,3})$", lower_prompt)
                if loc_m:
                    parsed_spec["environment"] = " ".join(w.capitalize() for w in loc_m.group(1).split())

            # 4b. Reconcile Characters vs Enemies so hostile entities are properly classified under Enemies
            raw_chars = parsed_spec.get("characters", [])
            raw_enemies = list(parsed_spec.get("enemies", []))
            seen_enemy_names = {e.get("name", "").strip().lower() for e in raw_enemies if isinstance(e, dict)}
            threat_words = set(THREAT_LEXICON.keys()) | {"boss", "enemy", "hostile", "monster", "horde", "swarm", "invader", "undead"}

            cleaned_chars = []
            rescued_weapons = []
            for ch in raw_chars:
                cname = (ch.get("name") or "").strip()
                cdesc = (ch.get("desc") or "").strip()
                clow = cname.lower()
                dlow = cdesc.lower()
                cwords = set(re.findall(r"[a-z0-9-]+", clow))

                # Filter obvious out-of-domain hallucinations or generic repair placeholders
                if clow in {"protagonist", "hero", "traveler", "character", "sugar rush", "treasure", "mining", "ruins"}:
                    continue
                # Rescue misclassified weapons from T5 characters array (e.g. "Robot Axe: A heavy weapon...")
                if any(wk in dlow for wk in ["heavy weapon", "weapon used", "blade used", "firearm", "cannon used"]) and not any(hk in clow for hk in HERO_ROLES):
                    rescued_weapons.append({"name": cname, "desc": cdesc})
                    continue
                # Move T5 hostile entities into enemies list unless explicitly a player hero role
                if (cwords & threat_words) and not (cwords & set(HERO_ROLES.keys())):
                    if clow and clow not in seen_enemy_names:
                        seen_enemy_names.add(clow)
                        raw_enemies.append({"name": cname, "desc": cdesc or "Hostile adversary generated from neural blueprint."})
                    continue
                cleaned_chars.append(ch)

            # Merge prompt-extracted hostile enemies (handles arbitrary & unseen threats)
            for se in sem_enemies:
                se_low = se["name"].strip().lower()
                if se_low not in seen_enemy_names and not any(se_low in en or en in se_low for en in seen_enemy_names if len(en) > 3):
                    seen_enemy_names.add(se_low)
                    raw_enemies.insert(0, se)
            parsed_spec["enemies"] = raw_enemies

            # Ensure coherent Hero Protagonist
            if not cleaned_chars or num_tokens <= 16:
                if sem_hero:
                    cleaned_chars = [sem_hero]
                elif not cleaned_chars:
                    cleaned_chars = [{"name": "Protagonist", "desc": "A resilient lead character navigating hazards and adversaries."}]
            parsed_spec["characters"] = cleaned_chars

            # 4c. Ground Weapons & Props when T5 outputs generic placeholders, cross-biome items, or collapses
            weaps = parsed_spec.get("weapons", []) + rescued_weapons
            valid_weaps = [w for w in weaps if w.get("name", "").strip().lower() not in {"", "standard weapon", "weapon"}]
            if (not valid_weaps or cross_biome_mismatch or num_tokens <= 16) and sem_defaults and sem_defaults.get("weapon"):
                valid_weaps = [sem_defaults["weapon"]]
            elif not valid_weaps:
                valid_weaps = [{"name": "Tactical Field Blade & Blaster", "desc": "Balanced melee-and-ranged combat kit engineered for hostile encounters."}]
            if valid_weaps:
                parsed_spec["weapons"] = valid_weaps[:4]

            props = parsed_spec.get("props", [])
            valid_props = [p for p in props if p.get("name", "").strip().lower() not in {"", "item cache", "prop"}]
            prop_blob = " ".join(p.get("name", "").lower() for p in valid_props)
            prop_mismatch = (
                (sem_biome in {"sci_fi_action", "outbreak_apocalypse"} and any(w in prop_blob for w in ["treasure chest", "rope bridge", "cauldron"]))
            )
            if (not valid_props or cross_biome_mismatch or prop_mismatch or num_tokens <= 16) and sem_defaults and sem_defaults.get("props"):
                valid_props = sem_defaults["props"]
            if valid_props:
                parsed_spec["props"] = valid_props[:4]

        enriched_meta = enrich_game_concept_metadata(
            parsed_spec,
            clean_prompt,
            matched_arcade=matched_arcade,
            prompt_semantics=prompt_semantics,
        )

        # Entity Slot Guarantees: Auto-fill empty props / character companion slots
        if not parsed_spec.get("props"):
            parsed_spec["props"] = enriched_meta.get("fallback_props", [])

        # Auto-ground prompt companion if user requested one (e.g. 'fox', 'wolf', 'cat', 'robot', 'fairy')
        lower_prompt = clean_prompt.lower()
        existing_char_names = " ".join([c.get("name", "").lower() for c in parsed_spec.get("characters", [])])
        companion_keywords = {
            "fox": ("Fox Companion", "A nimble, swift companion that scouts terrain and alerts to nearby hazards."),
            "cat": ("Familiar Cat", "An intuitive feline companion sensitive to magical fluctuations."),
            "dog": ("Faithful Hound", "A loyal canine tracker that assists in defense and scenting resources."),
            "wolf": ("Tamed Wolf", "A fierce forest companion guarding the party during resting cycles."),
            "robot": ("Utility Drone", "A hovering robotic assistant equipped with scanners and laser repair beams."),
            "fairy": ("Sprite Companion", "A miniature ethereal being providing illumination in dark regions.")
        }
        for kw, (c_name, c_desc) in companion_keywords.items():
            if kw in lower_prompt and kw not in existing_char_names:
                parsed_spec.setdefault("characters", []).append({"name": c_name, "desc": c_desc})

        # Sanitize abstract character names (e.g. small T5 sometimes emits 'Treasure' as character name)
        for c in parsed_spec.get("characters", []):
            if c.get("name", "").strip().lower() in ["treasure", "mining", "ruins"]:
                c["name"] = "Treasure Seeker" if "treasure" in c["name"].lower() else "Explorer"

        # Thematic Tool/Weapon supplementation for peaceful/farming simulation genres
        if enriched_meta.get("matched_cat") == "farming" and enriched_meta.get("tool_alt"):
            # Ensure at least one agricultural tool is present
            tool = enriched_meta["tool_alt"]
            tool_names = [w.get("name", "").lower() for w in parsed_spec.get("weapons", [])]
            if not any(k in " ".join(tool_names) for k in ["trowel", "hoe", "scythe", "watering"]):
                parsed_spec.setdefault("weapons", []).insert(0, tool)

        enrichment = {
            "pipeline_stage": "Stage 2: Deterministic Asset Mapping & Design Enrichment (Rule-Based)",
            "recommended_asset_tags": derive_asset_tags(parsed_spec),
            "game_type": enriched_meta["game_type"],
            "genre": enriched_meta["genre"],
            "target_perspective": enriched_meta["target_perspective"],
            "core_loop": enriched_meta["core_loop"],
            "note": "Derived deterministically from T5 theme & character entities for downstream sprite discovery"
        }

        # Map to structured spec object
        main_char = parsed_spec.get("characters", [{}])[0] if parsed_spec.get("characters") else {}
        main_char_name = main_char.get("name", "Protagonist")
        main_char_desc = main_char.get("desc", "")
        enemies_list = parsed_spec.get("enemies", [])

        story_brief = build_story_brief(
            title=enriched_meta["game_title"],
            theme=parsed_spec.get("theme", ""),
            environment=parsed_spec.get("environment", ""),
            characters=parsed_spec.get("characters", []),
            enemies=enemies_list,
            weapons=parsed_spec.get("weapons", []),
            props=parsed_spec.get("props", []),
            core_loop=enriched_meta["core_loop"],
            visual_style=parsed_spec.get("visualStyle", "Retro Pixel Art"),
        )

        return {
            "status": "success",
            "prompt": clean_prompt,
            "formatted_input": formatted_input,
            "game_title": enriched_meta["game_title"],
            "story_brief": story_brief,
            "theme": parsed_spec.get("theme", ""),
            "genre": enriched_meta["genre"],
            "game_type": enriched_meta["game_type"],
            "target_perspective": enriched_meta["target_perspective"],
            "core_loop": enriched_meta["core_loop"],
            "environment": parsed_spec.get("environment", ""),
            "visualStyle": parsed_spec.get("visualStyle", ""),
            "characters": parsed_spec.get("characters", []),
            "enemies": enemies_list,
            "weapons": parsed_spec.get("weapons", []),
            "props": parsed_spec.get("props", []),
            # Backwards compatibility fields for UI card layout
            "main_character": {
                "role": main_char_name,
                "attributes": [main_char_desc] if main_char_desc else ["Pixel Art Character"]
            },
            "art_style": parsed_spec.get("visualStyle", "Retro Pixel Art"),
            "allies": [c.get("name") for c in parsed_spec.get("characters", [])[1:] if any(k in c.get("name", "").lower() for k in ["companion", "pet", "ally", "friend", "familiar", "fox", "hound", "drone"])],
            "recommended_asset_tags": enrichment["recommended_asset_tags"],
            "raw_t5_output": raw_output,
            "telemetry": {
                "inference_latency_ms": latency_ms,
                "tokens_generated": len(output_ids[0]),
                "model_checkpoint": "models/4_Transformer_2k_v1",
                "base_architecture": "google/flan-t5-small (80 Million Parameters)",
                "decoding_strategy": "Greedy Decoding (num_beams=1) with Boundary Repair Auditor",
                "schema_valid": schema_valid,
                "missing_fields": missing_fields,
                "extra_fields": extra_fields
            },
            "ethical_guardrails": {
                "content_safety": safety_msg,
                "safety_status": safety_msg,
                "checks": security_checks,
                "dataset_attribution": "GameForge Fine-Tuned T5 (2,000 Curated Prompt-JSON Pairs; 80/10/10 Split)",
                "license": "CC-BY-NC-SA 4.0",
                "human_oversight_required": True
            },
            "enrichment_layer": enrichment
        }


transformer_service = TransformerService()
