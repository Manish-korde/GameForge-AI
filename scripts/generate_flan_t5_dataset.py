import json
import os

examples = [
    # ----------------- 1. FANTASY (1-5) -----------------
    {
        "input": "A young mage exploring a crystalline cavern to restore an elemental shrine.",
        "target": {
            "game_title": "Aetherium Spire",
            "genre": "Top-Down Action RPG",
            "visual_style": "16-bit Vibrant Pixel Art",
            "core_loop": {
                "mechanic": "Elemental spell-weaving and mana management",
                "objective": "Ignite the four elemental braziers and awaken the shrine core",
                "hazards": ["Falling crystal stalactites", "Mana siphon geysers", "Shattered ice crevasses"]
            },
            "world_setting": {
                "environment": "Crystalline Depths of Mount Vael",
                "atmosphere": "Luminescent violet reflections and echoing cavern drops",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Lyra the Elementalist", "role": "Protagonist", "stats": {"hp": 90, "speed": "Fast"}, "abilities": ["Frost Barrier", "Fire Dash", "Arcane Burst"]},
                {"name": "Gorgon Crystalline Golem", "role": "Boss", "stats": {"hp": 450, "speed": "Slow"}, "abilities": ["Prismatic Shockwave", "Crystal Spike Volley"]}
            ],
            "weapons": [
                {"name": "Spire Wand", "type": "Magic Ranged", "mechanic": "Fires homing energy bolts that consume 5 mana"}
            ],
            "props": [
                {"name": "Elemental Conduit Pillar", "type": "Puzzle Activator", "mechanic": "Aligns beam refraction angles when struck with spells"}
            ]
        }
    },
    {
        "input": "A knight defending a besieged castle against waves of shadow creatures.",
        "target": {
            "game_title": "Bastion of Sunfall",
            "genre": "Castle Defense Hack and Slash",
            "visual_style": "32-bit Gothic Pixel Art",
            "core_loop": {
                "mechanic": "Shield parrying and directional melee cleaves",
                "objective": "Prevent siege breaches until the dawn bell rings",
                "hazards": ["Burning pitch cauldrons", "Collapsing battlements", "Shadow tendril rifts"]
            },
            "world_setting": {
                "environment": "High Keep Ramparts",
                "atmosphere": "Blood-red moonlight and torrential rain with lightning flashes",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Sir Roland of Sunfall", "role": "Protagonist", "stats": {"hp": 150, "speed": "Normal"}, "abilities": ["Solar Shield Slam", "Whirlwind Cleave"]},
                {"name": "Dusk Wraith Reaver", "role": "Enemy", "stats": {"hp": 80, "speed": "Fast"}, "abilities": ["Phasing Teleport", "Life Leach Claws"]}
            ],
            "weapons": [
                {"name": "Sunforged Broadsword", "type": "Two-Handed Melee", "mechanic": "Charges radiant heat on consecutive three-hit combos"},
                {"name": "Tower Aegis", "type": "Defensive Shield", "mechanic": "Deflects incoming projectiles into nearby siege attackers"}
            ],
            "props": [
                {"name": "Signal Beacon", "type": "Interactive Station", "mechanic": "Summons royal archer volley across current lane when lit"}
            ]
        }
    },
    {
        "input": "A mischievous goblin rogue raiding an ancient elven vault for enchanted relics.",
        "target": {
            "game_title": "Gilded Gremlin",
            "genre": "Stealth Platformer",
            "visual_style": "Hand-Drawn Stylized 2D",
            "core_loop": {
                "mechanic": "Shadow-skulking, wall climbing, and lock picking",
                "objective": "Pocket the Crown of Stars and slip past the sanctum gates",
                "hazards": ["Pressure plate dart traps", "Enchanted detection wards", "Rotating mirror lasers"]
            },
            "world_setting": {
                "environment": "Sylvaen Gilded Vaults",
                "atmosphere": "Bioluminescent leaf glows with golden gilded arches",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Snark the Snatcher", "role": "Protagonist", "stats": {"hp": 70, "speed": "Very Fast"}, "abilities": ["Smoke Bomb", "Wall Cling", "Pocket Sand"]},
                {"name": "Arcane Sentinel", "role": "Enemy", "stats": {"hp": 120, "speed": "Normal"}, "abilities": ["Truesight Aura", "Glaive Swipe"]}
            ],
            "weapons": [
                {"name": "Rusty Hook Dagger", "type": "Light Melee", "mechanic": "Silent backstab dealing 4x critical stealth damage"}
            ],
            "props": [
                {"name": "Enchanted Relic Pedestal", "type": "Loot Container", "mechanic": "Triggers alarm lockdown unless replaced with equal-weight sandbag"}
            ]
        }
    },
    {
        "input": "An alchemist traversing a poisonous fungal bog to synthesize a pandemic cure.",
        "target": {
            "game_title": "Myco-Apothecary",
            "genre": "Survival Crafting RPG",
            "visual_style": "Rich 16-bit Painterly Pixel Art",
            "core_loop": {
                "mechanic": "Spore harvesting, flask brewing, and terrain vaporization",
                "objective": "Distill the Panacea Elixir at the Heart Mushroom sanctum",
                "hazards": ["Corrosive swamp sludge", "Toxic spore clouds", "Explosive puffballs"]
            },
            "world_setting": {
                "environment": "Rotspore Murkland",
                "atmosphere": "Heavy sickly-green mist with fluorescent bioluminescent caps",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Corin the Herbalist", "role": "Protagonist", "stats": {"hp": 85, "speed": "Normal"}, "abilities": ["Antidote Mist", "Concoction Lob"]},
                {"name": "Broodmother Spore Crawler", "role": "Boss", "stats": {"hp": 520, "speed": "Slow"}, "abilities": ["Acidic Vomit", "Swarm Hatching"]}
            ],
            "weapons": [
                {"name": "Glass Flask Mortar", "type": "Ranged Chemistry", "mechanic": "Lobs elemental reagents that neutralize toxic terrain hazards"}
            ],
            "props": [
                {"name": "Centuries-Old Cauldron", "type": "Crafting Station", "mechanic": "Refines raw spores into vital antidote injectors and buff tonics"}
            ]
        }
    },
    {
        "input": "A blind monk wielding resonance bells to battle mythical yokai spirits in mountain shrines.",
        "target": {
            "game_title": "Resonance: Echoes of Koya",
            "genre": "Isometric Rhythm Action",
            "visual_style": "Monochromatic Ink Wash Pixel Art",
            "core_loop": {
                "mechanic": "Soundwave echolocation and rhythmic bell strikes",
                "objective": "Cleanse the Torii shrines along the sacred pilgrim summit",
                "hazards": ["Spirit vortex rifts", "Crumbling stone bridges", "Ghostly wind shears"]
            },
            "world_setting": {
                "environment": "Fog-Bound Sacred Peak",
                "atmosphere": "Ethereal gray ink fog punctuated by striking vermilion lanterns",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Master Genkei", "role": "Protagonist", "stats": {"hp": 110, "speed": "Fast"}, "abilities": ["Sonar Pulse", "Sonic Deflection", "Echo Dash"]},
                {"name": "Oni of Hollow Bells", "role": "Boss", "stats": {"hp": 480, "speed": "Normal"}, "abilities": ["Discordant Roar", "Iron Kanabo Slam"]}
            ],
            "weapons": [
                {"name": "Bronze Chime Staff", "type": "Resonance Melee", "mechanic": "Unleashes directional vibration waves on beat-timed strikes"}
            ],
            "props": [
                {"name": "Great Temple Bonsho", "type": "Sonic Purifier", "mechanic": "Banishes spirit illusions in a wide radial zone when struck"}
            ]
        }
    },

    # ----------------- 2. SCI-FI (6-10) -----------------
    {
        "input": "A lone salvage engineer exploring a derelict generation ship adrift in deep space.",
        "target": {
            "game_title": "Zero Kelvin Drift",
            "genre": "Atmospheric 2D Sci-Fi Survival",
            "visual_style": "Moody Hi-Bit Pixel Art",
            "core_loop": {
                "mechanic": "Zero-G thruster navigation and airlock hacking",
                "objective": "Restore the cryogenic backup reactors and evacuate the black box",
                "hazards": ["Hull decompression vents", "Exposed plasma conduits", "Unstable zero-G gravity shifts"]
            },
            "world_setting": {
                "environment": "USS Horizon Derelict Hangar",
                "atmosphere": "Deep vacuum silence, flashing amber warning sirens, and drifting frost",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Engineer Vance", "role": "Protagonist", "stats": {"hp": 100, "speed": "Normal"}, "abilities": ["EVA Booster Burst", "Circuit Splicer"]},
                {"name": "Rogue Maintenance Drone", "role": "Enemy", "stats": {"hp": 60, "speed": "Fast"}, "abilities": ["Laser Arc Burn", "Self-Destruct Rush"]}
            ],
            "weapons": [
                {"name": "Plasma Arc Welder", "type": "Tool/CQC Weapon", "mechanic": "Melts blast door locks and severs robotic chassis joints"}
            ],
            "props": [
                {"name": "Emergency Oxygen Station", "type": "Life Support", "mechanic": "Refills EVA suit reserve tanks before asphyxiation timer expires"}
            ]
        }
    },
    {
        "input": "Fast-paced combat rover pilot fighting alien swarms across the surface of Mars.",
        "target": {
            "game_title": "Red Planet Raider",
            "genre": "Top-Down Twin-Stick Shooter",
            "visual_style": "Gritty Cyberpunk 16-bit",
            "core_loop": {
                "mechanic": "High-speed drifting and omnidirectional turret firing",
                "objective": "Secure terraforming pylons and exterminate chitinous broods",
                "hazards": ["Dust devil twisters", "Acid geyser craters", "Subterranean sand-worm tremors"]
            },
            "world_setting": {
                "environment": "Valles Marineris Basin",
                "atmosphere": "Rust-orange sky with violent sandstorms and crimson dust trails",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Captain Jax Briggs", "role": "Protagonist", "stats": {"hp": 200, "speed": "Very Fast"}, "abilities": ["Nitro Boost", "Deployable Minefield"]},
                {"name": "Chitin Brood Behemoth", "role": "Boss", "stats": {"hp": 650, "speed": "Fast"}, "abilities": ["Burrow Ambush", "Corrosive Slag Cannon"]}
            ],
            "weapons": [
                {"name": "Twin Rotary Gauss Cannons", "type": "Vehicle Ballistic", "mechanic": "High fire rate weapon that shreds armor at point-blank range"},
                {"name": "Heat-Seeking Micro-Missiles", "type": "Heavy Ordnance", "mechanic": "Fires a swarm of six targeting alien clusters"}
            ],
            "props": [
                {"name": "Solar Array Node", "type": "Power Depot", "mechanic": "Recharges rover kinetic shield when idling inside its radius"}
            ]
        }
    },
    {
        "input": "A space bounty hunter hunting rogue androids through neon orbital stations.",
        "target": {
            "game_title": "Apex Bounty: Orbital Drift",
            "genre": "2D Action Platformer",
            "visual_style": "Futuristic Neon Pixel Art",
            "core_loop": {
                "mechanic": "Grappling hook swinging and electromagnetic slug-shooting",
                "objective": "Locate and capture or neutralize the android ringleader",
                "hazards": ["High-voltage floor grids", "Automated laser sentries", "Breached glass decompression"]
            },
            "world_setting": {
                "environment": "Neo-Babylon Orbital Ring",
                "atmosphere": "Deep black space backdrop lit by pulsing magenta neon billboards",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Kallum Vex", "role": "Protagonist", "stats": {"hp": 110, "speed": "Fast"}, "abilities": ["Mag-Hook Grapple", "Holo-Decoy"]},
                {"name": "Unit X-9 Zero", "role": "Boss", "stats": {"hp": 400, "speed": "Very Fast"}, "abilities": ["Overclocked Slash", "EMP Shockwave"]}
            ],
            "weapons": [
                {"name": "Ion Scatter Carbine", "type": "Energy Ranged", "mechanic": "Disrupts cybernetic shields and stuns android chassis"}
            ],
            "props": [
                {"name": "Security Data Terminal", "type": "Intel Hack", "mechanic": "Reveals enemy patrol paths and disables regional cameras"}
            ]
        }
    },
    {
        "input": "Colonist terraforming an ice moon while managing geothermal power grids.",
        "target": {
            "game_title": "Cryo-Genesis",
            "genre": "Isometric Colony Builder / Action",
            "visual_style": "Clean Vector Isometric Pixel Art",
            "core_loop": {
                "mechanic": "Pipe-laying, heat modulation, and creature repelling",
                "objective": "Melt the permafrost core to activate the planetary biosphere dome",
                "hazards": ["Flash-freeze blizzards", "Cryo-volcanic fissures", "Predatory frost mantas"]
            },
            "world_setting": {
                "environment": "Europa Borealis Plain",
                "atmosphere": "Pale blue glaciers beneath the towering ringed shadow of Jupiter",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Pioneer Elena Frost", "role": "Protagonist", "stats": {"hp": 100, "speed": "Normal"}, "abilities": ["Thermal Shield", "Geothermal Harpoon"]},
                {"name": "Frost Burrower", "role": "Enemy", "stats": {"hp": 75, "speed": "Fast"}, "abilities": ["Glacial Pounce", "Freezing Mist"]}
            ],
            "weapons": [
                {"name": "Thermal Lance", "type": "Industrial Laser", "mechanic": "Melts thick ice blocks and cauterizes alien tissue"}
            ],
            "props": [
                {"name": "Geothermal Heat Regulator", "type": "Infrastructure", "mechanic": "Generates a radiating circle of warmth that sustains human health"}
            ]
        }
    },
    {
        "input": "Sub-atomic nanotechnology robot battling computer viruses inside a quantum mainframe.",
        "target": {
            "game_title": "Nano-Purge: Sector 0",
            "genre": "Twin-Stick Roguelite",
            "visual_style": "Vector Wireframe Pixel Hybrid",
            "core_loop": {
                "mechanic": "Bit-shifting dashes and logic gate manipulation",
                "objective": "Quarantine corrupted memory sectors and reboot the primary kernel",
                "hazards": ["Overflow memory spikes", "Bit-rot corruption tiles", "Firewall sweep beams"]
            },
            "world_setting": {
                "environment": "Quantum Silicon Core",
                "atmosphere": "Deep cobalt circuit boards glowing with brilliant golden data packets",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Unit Byte-1", "role": "Protagonist", "stats": {"hp": 80, "speed": "Very Fast"}, "abilities": ["Parity Shift", "Kernel Shield"]},
                {"name": "Trojan Dreadworm", "role": "Boss", "stats": {"hp": 500, "speed": "Normal"}, "abilities": ["Packet Flood", "Memory Leak Aura"]}
            ],
            "weapons": [
                {"name": "Logic Siphon Beam", "type": "Energy Projector", "mechanic": "Inverts infected code blocks into friendly protective barrier nodes"}
            ],
            "props": [
                {"name": "System Cache Gateway", "type": "Checkpoint Station", "mechanic": "Defragments current health status and uploads acquired subroutines"}
            ]
        }
    },

    # ----------------- 3. SURVIVAL (11-15) -----------------
    {
        "input": "A shipwrecked sailor surviving on a cursed archipelago haunted by sea phantoms.",
        "target": {
            "game_title": "Tides of Dagon",
            "genre": "Open-World Top-Down Survival",
            "visual_style": "Weathered 16-bit Pixel Art",
            "core_loop": {
                "mechanic": "Flotsam scavenging, raft construction, and sanity preservation",
                "objective": "Craft an ocean-going skiff and break the siren storm barrier",
                "hazards": ["Rising midnight tides", "Rotting shipwreck spikes", "Whispering siren mists"]
            },
            "world_setting": {
                "environment": "Bleak Bone Reef",
                "atmosphere": "Crashing gray waves, sea spray, and faint phosphorescent tide pools",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Castaway Jonah", "role": "Protagonist", "stats": {"hp": 100, "speed": "Normal"}, "abilities": ["Flint Spark", "Emergency Dive"]},
                {"name": "Barnacled Drowned", "role": "Enemy", "stats": {"hp": 65, "speed": "Slow"}, "abilities": ["Anchor Drag", "Saltwater Spew"]}
            ],
            "weapons": [
                {"name": "Hardened Whalebone Harpoon", "type": "Piercing Melee", "mechanic": "Impales aquatic enemies and pulls floating debris toward player"}
            ],
            "props": [
                {"name": "Driftwood Bonfire", "type": "Sanity Anchor", "mechanic": "Restores sanity level and cooks raw fish when fed dry timber"}
            ]
        }
    },
    {
        "input": "Arctic expedition survivor battling sub-zero temperatures and starving wolves in Siberia.",
        "target": {
            "game_title": "Whiteout: Zero Degrees",
            "genre": "Hardcore Survival Simulator",
            "visual_style": "Bleak Minimalist Pixel Art",
            "core_loop": {
                "mechanic": "Calorie burning management, shelter building, and campfire stoking",
                "objective": "Reach the abandoned radio weather station 20 kilometers north",
                "hazards": ["Hypothermic blizzards", "Thin frozen lake ice", "Rabid timber wolf packs"]
            },
            "world_setting": {
                "environment": "Siberian Taiga Wilderness",
                "atmosphere": "Biting crosswinds, crunching powder snow, and gray overcast twilight",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Dr. Sarah Lind", "role": "Protagonist", "stats": {"hp": 95, "speed": "Slow"}, "abilities": ["Snowshoe Sprint", "Triage Dressing"]},
                {"name": "Alpha Dire Wolf", "role": "Enemy", "stats": {"hp": 110, "speed": "Very Fast"}, "abilities": ["Pack Howl", "Tendon Bite"]}
            ],
            "weapons": [
                {"name": "Bolt-Action Hunting Rifle", "type": "Ballistic Longarm", "mechanic": "High single-shot damage with scarce ammo that alerts predators"}
            ],
            "props": [
                {"name": "Improvised Pine Bough Shelter", "type": "Rest Haven", "mechanic": "Provides windbreak protection and allows sleep without freezing"}
            ]
        }
    },
    {
        "input": "Desert nomad seeking subterranean oases while dodging monstrous dune worms.",
        "target": {
            "game_title": "Sands of the Maw",
            "genre": "Isometric Desert Survival",
            "visual_style": "Golden Warm Pixel Art",
            "core_loop": {
                "mechanic": "Hydration tracking, silent sand stepping, and solar navigation",
                "objective": "Fill water vessels and reach the Sanctuary Citadel of Ur",
                "hazards": ["Blistering midday heatstroke", "Sinkhole quicksand", "Sand-worm sound vibrations"]
            },
            "world_setting": {
                "environment": "The Bleached Erg",
                "atmosphere": "Blinding sunlight, shifting golden mirages, and hot arid winds",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Nomad Tariq", "role": "Protagonist", "stats": {"hp": 100, "speed": "Fast"}, "abilities": ["Sand Skim", "Rope Grapple"]},
                {"name": "Maw of the Erg", "role": "Boss", "stats": {"hp": 700, "speed": "Very Fast"}, "abilities": ["Vibration Breach", "Sand Vortex"]}
            ],
            "weapons": [
                {"name": "Curved Scimitar", "type": "Slashing Melee", "mechanic": "Lightweight blade suited for swift counter-attacks against raiders"}
            ],
            "props": [
                {"name": "Deep Desert Dew Collector", "type": "Resource Well", "mechanic": "Gathers condensation during cold desert night cycles"}
            ]
        }
    },
    {
        "input": "Subterranean miner trapped in a collapsed coal mine with dwindling torchlight and rising gas.",
        "target": {
            "game_title": "Choke Damp",
            "genre": "2D Claustrophobic Survival Horror",
            "visual_style": "High-Contrast Charcoal Pixel Art",
            "core_loop": {
                "mechanic": "Pickaxe tunneling, air pocket detection, and lantern fuel economy",
                "objective": "Excavate an ascent shaft to reach the surface ventilation flue",
                "hazards": ["Combustible methane gas pockets", "Ceiling cave-ins", "Flooded drainage tunnels"]
            },
            "world_setting": {
                "environment": "Blackwood Seam Shaft 4",
                "atmosphere": "Pitch-black darkness illuminated only by the faint orange glow of a carbide lamp",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Miner Cole", "role": "Protagonist", "stats": {"hp": 105, "speed": "Normal"}, "abilities": ["Pickaxe Pave", "Canary Cage Check"]},
                {"name": "Blind Cave Lurker", "role": "Enemy", "stats": {"hp": 70, "speed": "Fast"}, "abilities": ["Echolocation Screech", "Ambush Lunge"]}
            ],
            "weapons": [
                {"name": "Heavy Steel Mattock", "type": "Mining Melee", "mechanic": "Breaks brittle stone walls and crunches carapace armor"}
            ],
            "props": [
                {"name": "Hand-Crank Oxygen Bellows", "type": "Air Supply", "mechanic": "Pumps fresh air into localized pockets to clear dangerous gas"}
            ]
        }
    },
    {
        "input": "Post-apocalyptic scavenger building a fortified junkyard base against feral raiders.",
        "target": {
            "game_title": "Rust Haven",
            "genre": "Top-Down Base Defense Survival",
            "visual_style": "Grungy 16-bit Scrap Pixel Art",
            "core_loop": {
                "mechanic": "Scrap welding, barricade mounting, and perimeter defense",
                "objective": "Protect the water purification cistern until the rescue convoy arrives",
                "hazards": ["Corrosive acid rain puddles", "Exploding barrel chain reactions", "Rusty barbed wire"]
            },
            "world_setting": {
                "environment": "Wasteland Scrap Yard",
                "atmosphere": "Smog-choked amber dusk with smoke rising from burning tire piles",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Rusty Miller", "role": "Protagonist", "stats": {"hp": 120, "speed": "Normal"}, "abilities": ["Welder Repair", "Turret Deploy"]},
                {"name": "Scrap-Ripper Warlord", "role": "Boss", "stats": {"hp": 480, "speed": "Fast"}, "abilities": ["Chainsaw Charge", "Molotov Barrage"]}
            ],
            "weapons": [
                {"name": "Pneumatic Nail Cannon", "type": "Improvised Ranged", "mechanic": "Fires high-velocity industrial nails that pin light enemies to walls"}
            ],
            "props": [
                {"name": "Diesel Water Purifier", "type": "Defense Objective", "mechanic": "Produces consumable clean water bottles as long as fuel is supplied"}
            ]
        }
    },

    # ----------------- 4. HORROR (16-20) -----------------
    {
        "input": "An asylum night watchman investigating anomalous whispering hallways with only a failing flashlight.",
        "target": {
            "game_title": "Sanitarium 1904",
            "genre": "Psychological Survival Horror",
            "visual_style": "Dithered Retro 32-bit Horror Art",
            "core_loop": {
                "mechanic": "Battery conservation, breath holding, and audio tracking",
                "objective": "Collect warden keycards and locate the basement isolation ward",
                "hazards": ["Flickering strobe halls", "Hallucinatory shadow hands", "Glass-littered floors"]
            },
            "world_setting": {
                "environment": "St. Jude Psychiatric Hospital",
                "atmosphere": "Peeling green wallpaper, flickering incandescent bulbs, and distant screams",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Guard Arthur Bell", "role": "Protagonist", "stats": {"hp": 80, "speed": "Normal"}, "abilities": ["Focus Beam", "Quiet Step"]},
                {"name": "The Restrained One", "role": "Boss", "stats": {"hp": 350, "speed": "Very Fast"}, "abilities": ["Straitjacket Rush", "Psychic Screech"]}
            ],
            "weapons": [
                {"name": "Electric Stun Baton", "type": "Non-Lethal Defense", "mechanic": "Discharges battery power to temporarily paralyze pursuing apparitions"}
            ],
            "props": [
                {"name": "Wall-Mounted Fuse Box", "type": "Circuit Puzzle", "mechanic": "Restores emergency lighting in adjacent corridors to repel shadows"}
            ]
        }
    },
    {
        "input": "A folkloric investigator exploring an abandoned Romanian village stalked by strigoi vampires.",
        "target": {
            "game_title": "Strigoi: The Black Wood",
            "genre": "Side-Scrolling Gothic Horror",
            "visual_style": "Dark Castlevania-Inspired Pixel Art",
            "core_loop": {
                "mechanic": "Crossbow reloading, garlic warding, and stake driving",
                "objective": "Consecrate the desecrated village cemetery church",
                "hazards": ["Spike pit graves", "Hanging iron gibbets", "Mist apparition traps"]
            },
            "world_setting": {
                "environment": "Village of Borgo Pass",
                "atmosphere": "Freezing mountain wind, full harvest moon, and howling distant wolves",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Father Lucian", "role": "Protagonist", "stats": {"hp": 100, "speed": "Normal"}, "abilities": ["Holy Water Splash", "Crucifix Ward"]},
                {"name": "Elder Strigoi Noble", "role": "Boss", "stats": {"hp": 420, "speed": "Fast"}, "abilities": ["Bat Swarm Disperse", "Blood Drain Leap"]}
            ],
            "weapons": [
                {"name": "Silver-Tipped Stake Crossbow", "type": "Piercing Ranged", "mechanic": "Instantly executes stunned undead when struck in the heart"}
            ],
            "props": [
                {"name": "Cenotaph Altar", "type": "Consecration Node", "mechanic": "Provides a temporary safe zone where blood-drinking fiends cannot enter"}
            ]
        }
    },
    {
        "input": "Deep sea research diver fleeing eldritch abominations in an oceanic trench station.",
        "target": {
            "game_title": "Abyssal Maw",
            "genre": "Underwater Horror Exploration",
            "visual_style": "Deep Oceanic Atmospheric Pixel Art",
            "core_loop": {
                "mechanic": "Sonar pinging, oxygen regulation, and thermal flare throwing",
                "objective": "Ascend to the decompression diving bell before pressure collapse",
                "hazards": ["Crushing water pressure breaches", "Predatory angler lures", "Hypothermal deep currents"]
            },
            "world_setting": {
                "environment": "Hadal Research Platform Hadfield",
                "atmosphere": "Ink-black water with blinding bioluminescent flashes and metal groaning",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Diver Noah James", "role": "Protagonist", "stats": {"hp": 90, "speed": "Slow"}, "abilities": ["Sonar Beacon", "Emergency Buoyancy Lift"]},
                {"name": "The Trench Leviathan", "role": "Boss", "stats": {"hp": 600, "speed": "Fast"}, "abilities": ["Bioluminescent Hypnosis", "Abyssal Tail Sweep"]}
            ],
            "weapons": [
                {"name": "Pneumatic Harpoon Gun", "type": "Aquatic Ballistic", "mechanic": "Tethers large aquatic monstrosities to heavy anchor points"}
            ],
            "props": [
                {"name": "Pressurized Airlock Pod", "type": "Safe Chamber", "mechanic": "Drains seawater and restores suit atmospheric pressure"}
            ]
        }
    },
    {
        "input": "A Victorian coroner trapped inside a city morgue during an unnatural reanimation event.",
        "target": {
            "game_title": "Mortuary Midnight",
            "genre": "Isometric Survival Horror",
            "visual_style": "Sepia Muted Vintage Pixel Art",
            "core_loop": {
                "mechanic": "Dissection tool scavenging, salt barrier spreading, and furnace burning",
                "objective": "Incinerate the source corpse before midnight strikes twelve",
                "hazards": ["Slick embalming fluid spills", "Reanimated grasping limbs", "Escaping toxic formaldehydes"]
            },
            "world_setting": {
                "environment": "Whitechapel Public Mortuary",
                "atmosphere": "Damp cobblestone dampness, gaslight flickering, and heavy formaldehyde fumes",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Dr. Edwin Marsh", "role": "Protagonist", "stats": {"hp": 85, "speed": "Normal"}, "abilities": ["Formalin Splash", "Door Barricade"]},
                {"name": "The Stitchwork Cadaver", "role": "Boss", "stats": {"hp": 380, "speed": "Slow"}, "abilities": ["Meat Hook Cleave", "Cadaveric Stench"]}
            ],
            "weapons": [
                {"name": "Heavy Bone Saw", "type": "Surgical Melee", "mechanic": "Dismembers animated appendages to cripple enemy movement"}
            ],
            "props": [
                {"name": "Crematorium Furnace", "type": "Incineration Device", "mechanic": "Permanently destroys supernatural corpses to prevent reanimation"}
            ]
        }
    },
    {
        "input": "A hiker lost in the Black Forest encountering shifting trees and pagan effigies.",
        "target": {
            "game_title": "Wicker and Wood",
            "genre": "Top-Down Folk Horror",
            "visual_style": "Autumnal Earthy Pixel Art",
            "core_loop": {
                "mechanic": "Compass orienteering, lantern light maintenance, and effigy burning",
                "objective": "Find the forest boundary road before the blood moon rises",
                "hazards": ["Shifting briar mazes", "Pagan bone snare traps", "Disorienting fog walls"]
            },
            "world_setting": {
                "environment": "Odenwald Ancient Forest",
                "atmosphere": "Dead amber leaves, twisting gnarled pine trunks, and unnatural raven silence",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Hiker Daniel", "role": "Protagonist", "stats": {"hp": 90, "speed": "Fast"}, "abilities": ["Match Strike", "Sprint Vault"]},
                {"name": "The Antlered Husk", "role": "Boss", "stats": {"hp": 440, "speed": "Fast"}, "abilities": ["Root Strangle", "Primal Shriek"]}
            ],
            "weapons": [
                {"name": "Survival Hatchet", "type": "Woodland Tool", "mechanic": "Chops down blocking thorn barriers and defends against beasts"}
            ],
            "props": [
                {"name": "Wicker Totem", "type": "Curse Origin", "mechanic": "Burning this removes supernatural barrier fog from surrounding area"}
            ]
        }
    },

    # ----------------- 5. CYBERPUNK (21-25) -----------------
    {
        "input": "A street samurai infiltrating an Arasaka-style megacorp tower to extract stolen neural data.",
        "target": {
            "game_title": "Neon Ronin: Infiltration",
            "genre": "Fast-Paced Action Platformer",
            "visual_style": "Cyberpunk Synthwave Pixel Art",
            "core_loop": {
                "mechanic": "Wall running, bullet deflecting, and cyberware dash slicing",
                "objective": "Infiltrate floor 90 and copy the experimental neuro-chip",
                "hazards": ["Security laser grids", "Elevator shaft air-drops", "Thermal sensor auto-turrets"]
            },
            "world_setting": {
                "environment": "Kurosawa Heavy Industries Tower",
                "atmosphere": "Rain-slicked neon glass, holographic dragon ads, and deep synth bass",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Ren 'Zero' Kusanagi", "role": "Protagonist", "stats": {"hp": 100, "speed": "Very Fast"}, "abilities": ["Cyber-Dash", "Bullet Parry", "Overclock"]},
                {"name": "Corporate Enforcer Juggernaut", "role": "Boss", "stats": {"hp": 550, "speed": "Slow"}, "abilities": ["Minigun Sweep", "Riot Shield Charge"]}
            ],
            "weapons": [
                {"name": "Monofilament Katana", "type": "High-Frequency Blade", "mechanic": "Slices through metal drone plating and reflects kinetic ammunition"}
            ],
            "props": [
                {"name": "Terminal Access Node", "type": "Cyberdeck Station", "mechanic": "Triggers local camera blackout and opens blast doors on hack"}
            ]
        }
    },
    {
        "input": "A drone pilot operating in the rainy alleyways of a dystopian megacity.",
        "target": {
            "game_title": "Sky-Grid Syndicate",
            "genre": "Top-Down Tactical Shooter",
            "visual_style": "Rain-Streaked Neon Dark Pixel Art",
            "core_loop": {
                "mechanic": "Dual drone coordination, security camera spoofing, and EMP blasting",
                "objective": "Disrupt police barricades and escort the rebel smuggler van",
                "hazards": ["Electrified neon signs", "Surveillance spotlight drones", "Puddle shock traps"]
            },
            "world_setting": {
                "environment": "Sector 4 Chinatown Slums",
                "atmosphere": "Perpetual acid drizzle, flickering neon noodles signs, and steam vents",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Hacker Felix", "role": "Protagonist", "stats": {"hp": 85, "speed": "Fast"}, "abilities": ["Deploy Recon Drone", "EMP Grenade"]},
                {"name": "Psycho Police Mech", "role": "Boss", "stats": {"hp": 520, "speed": "Normal"}, "abilities": ["Cluster Rocket", "Shock Tether"]}
            ],
            "weapons": [
                {"name": "Suppressed Smart Pistol", "type": "Micro-Missile Sidearm", "mechanic": "Locks onto highlighted cybernetic implants with tracking rounds"}
            ],
            "props": [
                {"name": "Sub-Grid Junction Box", "type": "Power Relay", "mechanic": "Overloads power in the sector, causing short circuits to nearby foes"}
            ]
        }
    },
    {
        "input": "An underground street racer upgrading bionic limbs to compete in high-stakes death rallies.",
        "target": {
            "game_title": "Chrome and Nitro",
            "genre": "Combat Racing Action RPG",
            "visual_style": "16-bit Pseudo-3D Isometric Pixel Art",
            "core_loop": {
                "mechanic": "High-octane drifting, side-swipe ramming, and turret blasting",
                "objective": "Win the Cyber-Grand Prix and take out rival crime syndicate racers",
                "hazards": ["Spillway oil slicks", "Retractable highway spikes", "Laser toll barriers"]
            },
            "world_setting": {
                "environment": "Neo-Detroit Skyway Loop",
                "atmosphere": "Turbulent smog, blinding headlight streaks, and roaring combustion engines",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Racer Jax V", "role": "Protagonist", "stats": {"hp": 140, "speed": "Very Fast"}, "abilities": ["Nitrous Surge", "Adrenaline Reflexes"]},
                {"name": "Warlord Diesel", "role": "Boss", "stats": {"hp": 600, "speed": "Fast"}, "abilities": ["Flamethrower Exhaust", "Spike Wheel Shred"]}
            ],
            "weapons": [
                {"name": "Hood-Mounted Chainguns", "type": "Vehicle Armament", "mechanic": "Unleashes high-spread armor-piercing bullets at leading vehicles"}
            ],
            "props": [
                {"name": "Turbo Boost Pad", "type": "Track Booster", "mechanic": "Propels vehicles forward at 200% speed while granting collision immunity"}
            ]
        }
    },
    {
        "input": "A cyber-medic rescuing wounded civilians from a quarantine sector overrun by infected cyborgs.",
        "target": {
            "game_title": "Bio-Graft Protocol",
            "genre": "Top-Down Extraction Action",
            "visual_style": "Industrial Grimy Cyberpunk Pixel Art",
            "core_loop": {
                "mechanic": "Triage healing, civilian escorting, and automated barricading",
                "objective": "Evacuate survivors to the orbital medical dropship LZ",
                "hazards": ["Biohazard gas leaks", "Corrupted automated triage turrets", "Collapsing catwalks"]
            },
            "world_setting": {
                "environment": "Sector 9 Quarantine Ward",
                "atmosphere": "Emergency red strobes, bio-hazard warnings, and hiss of ruptured pipes",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Dr. Maya Chen", "role": "Protagonist", "stats": {"hp": 110, "speed": "Normal"}, "abilities": ["Medi-Spray Burst", "Nanite Stim Shot"]},
                {"name": "Aberrant Cyber-Carcass", "role": "Boss", "stats": {"hp": 460, "speed": "Fast"}, "abilities": ["Hydraulic Flail", "Viral Cloud Release"]}
            ],
            "weapons": [
                {"name": "Defibrillator Arc Rifle", "type": "Shock Weapon", "mechanic": "Arcs electrical current between mechanical enemies to fry circuits"}
            ],
            "props": [
                {"name": "Cryo-Stasis Chamber", "type": "Civilian Safe Pod", "mechanic": "Stabilizes critically injured survivors until final evacuation arrives"}
            ]
        }
    },
    {
        "input": "A black-market cyberware smuggler dealing with corrupt law enforcement in an underground bazaar.",
        "target": {
            "game_title": "Chrome Bazaar",
            "genre": "Stealth Trading RPG",
            "visual_style": "Dense Urban Isometric Pixel Art",
            "core_loop": {
                "mechanic": "Contraband bargaining, bribe distribution, and shadow evasion",
                "objective": "Smuggle military-grade neural chips out through the sewage canal",
                "hazards": ["K-9 Cyberhound sniffing units", "Drone scanning nets", "Security checkpoint gates"]
            },
            "world_setting": {
                "environment": "The Under-City Bazaar",
                "atmosphere": "Stifling humidity, smell of sizzling street food, and neon steam",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Kenji the Dealer", "role": "Protagonist", "stats": {"hp": 90, "speed": "Fast"}, "abilities": ["Chameleon Cloak", "Bribe Toss"]},
                {"name": "Inspector Kurtz", "role": "Enemy", "stats": {"hp": 150, "speed": "Normal"}, "abilities": ["Subpoena Taser", "Squad Call-In"]}
            ],
            "weapons": [
                {"name": "Concealed Wrist Dart", "type": "Silent Ranged", "mechanic": "Injects a tranquilizer neurotoxin for silent non-lethal takedowns"}
            ],
            "props": [
                {"name": "Fake Waste Bin", "type": "Stash Hiding Spot", "mechanic": "Allows player to conceal illicit inventory during spot police frisks"}
            ]
        }
    },

    # ----------------- 6. STEAMPUNK (26-30) -----------------
    {
        "input": "An airship captain battling sky pirates amid roaring steam engines and cloudy skies.",
        "target": {
            "game_title": "Aeronaut: Skies of Brass",
            "genre": "Side-Scrolling Airship Dogfighter",
            "visual_style": "Detailed Brass and Copper Pixel Art",
            "core_loop": {
                "mechanic": "Broadside cannon aiming, boiler pressure venting, and sky dodging",
                "objective": "Escort the Royal Zephyr past the Dread Pirate armada",
                "hazards": ["Thunderstorm lightning strikes", "Floating sky minefields", "Thermal updraft shears"]
            },
            "world_setting": {
                "environment": "The Cumulus Archipelago",
                "atmosphere": "Golden sunset clouds punctured by billows of black steam and burning embers",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Captain Arthur Sterling", "role": "Protagonist", "stats": {"hp": 180, "speed": "Normal"}, "abilities": ["Boiler Overdrive", "Harpoon Grapple"]},
                {"name": "Dread Captain Vane", "role": "Boss", "stats": {"hp": 650, "speed": "Fast"}, "abilities": ["Chain-Shot Volley", "Zeppelin Ramming"]}
            ],
            "weapons": [
                {"name": "Steam-Pressure Rotary Cannon", "type": "Airship Artillery", "mechanic": "Rapid fire that requires venting steam to prevent overheating"}
            ],
            "props": [
                {"name": "Coal Boiler Regulator", "type": "Ship Subsystem", "mechanic": "Shoveling coal boosts ship flight speed at the risk of boiler explosion"}
            ]
        }
    },
    {
        "input": "A clockwork clockmaker building mechanical automaton bodyguards to survive a factory revolt.",
        "target": {
            "game_title": "Cog and Crucible",
            "genre": "Top-Down Tactical Strategy / Action",
            "visual_style": "Sepia Tone Clockwork Pixel Art",
            "core_loop": {
                "mechanic": "Spring winding, gear swapping, and automaton command issuing",
                "objective": "Quell the rogue Steam-Foundry AI and shut down the master piston",
                "hazards": ["Molten brass cauldrons", "High-speed conveyor belts", "Crushing stamping presses"]
            },
            "world_setting": {
                "environment": "Grand Victoria Ironworks",
                "atmosphere": "Deafening metallic clatter, hissing steam geysers, and glowing smelting furnaces",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Horologist Barnaby", "role": "Protagonist", "stats": {"hp": 90, "speed": "Normal"}, "abilities": ["Clockwork Sentry Deploy", "Winding Key Boost"]},
                {"name": "Goliath Steam Golem", "role": "Boss", "stats": {"hp": 550, "speed": "Slow"}, "abilities": ["Steam Burst Cleave", "Piston Ground Pound"]}
            ],
            "weapons": [
                {"name": "Rivet Gun", "type": "Pneumatic Projector", "mechanic": "Fires red-hot industrial rivets that pierce through steel plating"}
            ],
            "props": [
                {"name": "Master Steam Valve", "type": "Environmental Lever", "mechanic": "Releases pressurized steam into the room to blind and damage attackers"}
            ]
        }
    },
    {
        "input": "A Victorian gentleman thief using brass gadgets to rob high-society zeppelins.",
        "target": {
            "game_title": "The Gilded Aeronaut",
            "genre": "Stealth Heist Adventure",
            "visual_style": "Elegant Victorian 16-bit Pixel Art",
            "core_loop": {
                "mechanic": "Lockpicking mechanism cracking, clockwork grapple climbing, and disguises",
                "objective": "Crack the Duke's pneumatic safe and parachute from the gala airship",
                "hazards": ["Mechanical bloodhound sniffers", "Watchman spotlights", "Fragile stained glass domes"]
            },
            "world_setting": {
                "environment": "The Imperial Sky-Gala Zeppelin",
                "atmosphere": "Polished mahogany wood, champagne crystal clinking, and velvet curtains",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Lord Percival Croft", "role": "Protagonist", "stats": {"hp": 85, "speed": "Fast"}, "abilities": ["Flash-Powder Cane", "Grapple Umbrella"]},
                {"name": "Chief Constable Ironclast", "role": "Enemy", "stats": {"hp": 130, "speed": "Normal"}, "abilities": ["Manacle Throw", "Baton Jab"]}
            ],
            "weapons": [
                {"name": "Pneumatic Dart Cane", "type": "Concealed Weapon", "mechanic": "Silent ranged takedowns that put socialite guards to sleep"}
            ],
            "props": [
                {"name": "Pneumatic Dial Safe", "type": "Puzzle Heist Target", "mechanic": "Requires listening to audio tumbler clicks to unlock diamond hoard"}
            ]
        }
    },
    {
        "input": "Subterranean steam-drill excavator digging through molten magma crust to unearth fossils.",
        "target": {
            "game_title": "Drill-Core Victoria",
            "genre": "Vertical Mining Action Roguelike",
            "visual_style": "Earthy Bronze and Molten Pixel Art",
            "core_loop": {
                "mechanic": "Directional drilling, water cooling, and ore collection",
                "objective": "Drill through 5,000 meters of subterranean bedrock to the core deposit",
                "hazards": ["Magma flow pockets", "Pressurized gas chambers", "Cave-dwelling rock squids"]
            },
            "world_setting": {
                "environment": "Mantle Crust Strata",
                "atmosphere": "Oppressive red magma glow, bubbling lava lakes, and grinding drill teeth",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Engineer Thaddeus", "role": "Protagonist", "stats": {"hp": 160, "speed": "Normal"}, "abilities": ["Hydraulic Turbo Drill", "Coolant Flush"]},
                {"name": "Obsidian Wyrm", "role": "Boss", "stats": {"hp": 580, "speed": "Fast"}, "abilities": ["Magma Spit", "Subterranean Breach"]}
            ],
            "weapons": [
                {"name": "Diamond-Tipped Industrial Auger", "type": "Drilling Weapon", "mechanic": "Grinds through heavy stone shields and shreds subterranean monsters"}
            ],
            "props": [
                {"name": "Coolant Reservoir Station", "type": "Thermal Well", "mechanic": "Replenishes ice-water canisters to cool overheated drill bits"}
            ]
        }
    },
    {
        "input": "A prosthetic-limbed vigilante fighting steam-baron enforcers in the polluted smog districts.",
        "target": {
            "game_title": "Iron Arm: Coalfall",
            "genre": "2D Beat-Em-Up / Brawler",
            "visual_style": "Dark Industrial Grimy 32-bit",
            "core_loop": {
                "mechanic": "Pneumatic punch combos, steam vent dodging, and crowd throws",
                "objective": "Dethrone the Coal Syndicate and liberate the slums' fresh air supply",
                "hazards": ["Bursting high-pressure pipes", "Electrified tram rails", "Vat slag spillage"]
            },
            "world_setting": {
                "environment": "The Soot District Slums",
                "atmosphere": "Thick brown smog, soot-caked brick buildings, and yellow gas lanterns",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Evelyn 'Steam-Fist' Gray", "role": "Protagonist", "stats": {"hp": 130, "speed": "Fast"}, "abilities": ["Piston Uppercut", "Steam Shockwave"]},
                {"name": "Baron Von Plackett", "role": "Boss", "stats": {"hp": 500, "speed": "Slow"}, "abilities": ["Mechanical Cane Gatling", "Slag Throw"]}
            ],
            "weapons": [
                {"name": "Steam-Piston Gauntlet", "type": "Prosthetic Brawling", "mechanic": "Releases explosive bursts of compressed steam on third heavy hit"}
            ],
            "props": [
                {"name": "Smog Filtration Vent", "type": "Air Purifier", "mechanic": "Temporarily clears toxic smog in the battle area to regenerate stamina"}
            ]
        }
    },

    # ----------------- 7. WESTERN (31-35) -----------------
    {
        "input": "A lone gunslinger hunting outlaws across desert ghost towns in the frontier.",
        "target": {
            "game_title": "Dead Dust Redemption",
            "genre": "Top-Down Tactical Western",
            "visual_style": "Dusty Sun-Baked 16-bit Pixel Art",
            "core_loop": {
                "mechanic": "Revolver fanning, cover vaulting, and quick-draw duel timing",
                "objective": "Eliminate the Black Hat Gang and collect the sheriff bounty",
                "hazards": ["Barbed wire fences", "Dynamite crates", "Scorpion infested rocks"]
            },
            "world_setting": {
                "environment": "Brimstone Gulch Ghost Town",
                "atmosphere": "Howling desert winds, tumbleweeds rolling down dusty wooden saloon roads",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "The Drifter", "role": "Protagonist", "stats": {"hp": 100, "speed": "Fast"}, "abilities": ["Dead-Eye Focus", "Combat Roll"]},
                {"name": "Black Hat Bart", "role": "Boss", "stats": {"hp": 380, "speed": "Normal"}, "abilities": ["Dual Peacemaker Fan", "Dynamite Toss"]}
            ],
            "weapons": [
                {"name": "Six-Shot Colt Peacemaker", "type": "Ballistic Handgun", "mechanic": "Can be fanned for 6 rapid bullets with increased recoil spread"}
            ],
            "props": [
                {"name": "Wooden Water Tower", "type": "Destructible Terrain", "mechanic": "Can be shot to collapse water and drown enemy cover positions"}
            ]
        }
    },
    {
        "input": "A mystical Native tracker hunting cursed cryptids in the haunted frontier prairies.",
        "target": {
            "game_title": "Spirits of the Badlands",
            "genre": "Isometric Supernatural Action RPG",
            "visual_style": "Atmospheric Hand-Crafted 2D",
            "core_loop": {
                "mechanic": "Animal tracking, bow drawing, and spirit totemic warding",
                "objective": "Hunt the Wendigo stalking the winter mountain pass",
                "hazards": ["Sudden blizzard whiteouts", "Frozen wolf traps", "Cursed animal totems"]
            },
            "world_setting": {
                "environment": "The Painted Badlands",
                "atmosphere": "Deep indigo night sky with blazing stars and dancing campfires",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Chayton Night-Bow", "role": "Protagonist", "stats": {"hp": 105, "speed": "Fast"}, "abilities": ["Eagle Eye Vision", "Spirit Wolf Summon"]},
                {"name": "The Prairie Wendigo", "role": "Boss", "stats": {"hp": 520, "speed": "Very Fast"}, "abilities": ["Flesh-Tear Lunge", "Howl of Frost"]}
            ],
            "weapons": [
                {"name": "Recurve Ash Bow", "type": "Archery Ranged", "mechanic": "Silent arrows infused with spirit herbs to stun supernatural foes"}
            ],
            "props": [
                {"name": "Sacred Cedar Smudge Pot", "type": "Ward Circle", "mechanic": "Repels cryptid beasts and prevents player from being ambushed while resting"}
            ]
        }
    },
    {
        "input": "Train engineer defending an armored locomotive against horseback bandits.",
        "target": {
            "game_title": "Iron Horse Defense",
            "genre": "Side-Scrolling Action Shooter",
            "visual_style": "High-Speed Kinetic Pixel Art",
            "core_loop": {
                "mechanic": "Car-to-car platforming, coal shoveling, and turret shooting",
                "objective": "Keep the train moving and deliver the federal gold shipment",
                "hazards": ["Low-clearance railway tunnels", "Uncoupled runaway boxcars", "Track sabotages"]
            },
            "world_setting": {
                "environment": "Great Continental Railroad",
                "atmosphere": "Golden prairie rushing by at high speed with billowing locomotive smoke",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Conductor Sean O'Malley", "role": "Protagonist", "stats": {"hp": 120, "speed": "Normal"}, "abilities": ["Wrench Swing", "Throttle Surge"]},
                {"name": "Bandit Leader 'Canyon Jack'", "role": "Boss", "stats": {"hp": 450, "speed": "Fast"}, "abilities": ["Horseback Lasso", "Lever-Action Volley"]}
            ],
            "weapons": [
                {"name": "Coachman Double-Barrel Shotgun", "type": "Scatter Gun", "mechanic": "Devastating close-range blast that knocks attackers off train roofs"}
            ],
            "props": [
                {"name": "Train Whistle Lever", "type": "Signaling Tool", "mechanic": "Disorients horseback bandits riding alongside the engine"}
            ]
        }
    },
    {
        "input": "An undead gunslinger crawling out of his grave seeking vengeance on his killer.",
        "target": {
            "game_title": "Gravebound: Dead Man's Gun",
            "genre": "Gothic Western Roguelite",
            "visual_style": "Monochromatic and Crimson Pixel Art",
            "core_loop": {
                "mechanic": "Bone-reloading, soul collecting, and demonic shadow stepping",
                "objective": "Defeat the corrupt preacher who betrayed you to the hangman",
                "hazards": ["Consecrated ground burn", "Graveyard thorny brambles", "Hanging noose traps"]
            },
            "world_setting": {
                "environment": "Boot Hill Cemetery",
                "atmosphere": "Eerie greenish graveyard fog beneath a sickly yellow sickle moon",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Revenant Cole", "role": "Protagonist", "stats": {"hp": 90, "speed": "Normal"}, "abilities": ["Grave Teleport", "Hellfire Fan"]},
                {"name": "Corrupt Preacher Kane", "role": "Boss", "stats": {"hp": 480, "speed": "Fast"}, "abilities": ["Brimstone Bible Blast", "Zealot Mob Summon"]}
            ],
            "weapons": [
                {"name": "Brimstone Schofield Revolver", "type": "Supernatural Firearms", "mechanic": "Shoots cursed fire bullets that consume harvested enemy souls"}
            ],
            "props": [
                {"name": "Desecrated Tombstone", "type": "Revival Beacon", "mechanic": "Allows player to resurrect once per level with half ammunition"}
            ]
        }
    },
    {
        "input": "Frontier sheriff defending an isolated jailhouse from a siege during a thunderstorm.",
        "target": {
            "game_title": "Lockdown at Fort Laramie",
            "genre": "Wave-Based Horde Survival",
            "visual_style": "Chiaroscuro Noir Western Pixel Art",
            "core_loop": {
                "mechanic": "Window boarding, ammo rationing, and precision marksmanship",
                "objective": "Hold the cellhouse until federal marshals arrive at daybreak",
                "hazards": ["Lightning-sparked roof fires", "Shattered window glass", "Breached rear cell walls"]
            },
            "world_setting": {
                "environment": "Town Jailhouse Bastion",
                "atmosphere": "Torrential rain drumming on tin roofs, flashes of lightning through blinds",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Sheriff Wyatt Graves", "role": "Protagonist", "stats": {"hp": 110, "speed": "Normal"}, "abilities": ["Quick Reload", "Barricade Hammer"]},
                {"name": "Outlaw Demolitionist 'Mad Dog'", "role": "Boss", "stats": {"hp": 420, "speed": "Slow"}, "abilities": ["Dynamite Bundle", "Shotgun Breach"]}
            ],
            "weapons": [
                {"name": "Winchester Model 1873", "type": "Lever-Action Rifle", "mechanic": "High-accuracy magazine rifle that pierces through wooden window frames"}
            ],
            "props": [
                {"name": "Heavy Iron Jail Cell Door", "type": "Fortification", "mechanic": "Can be locked to isolate and trap invading bandit clusters"}
            ]
        }
    },

    # ----------------- 8. MYSTERY (36-40) -----------------
    {
        "input": "A 1940s private eye solving a murder in a fog-drenched dockside warehouse.",
        "target": {
            "game_title": "Shadows on the Pier",
            "genre": "Noir Point-and-Click / Detective RPG",
            "visual_style": "16-bit Black and White Film Noir Pixel Art",
            "core_loop": {
                "mechanic": "Clue dusting, suspect interrogation, and deduction connection",
                "objective": "Identify the waterfront killer before the midnight ferry docks",
                "hazards": ["Slippery wet timber docks", "Armed mobster patrols", "Drowning tide swells"]
            },
            "world_setting": {
                "environment": "Pier 42 Maritime Warehouses",
                "atmosphere": "Thick coastal fog, yellow streetlamp cones, and sound of distant foghorns",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Detective Sam Spade Jr.", "role": "Protagonist", "stats": {"hp": 90, "speed": "Normal"}, "abilities": ["Magnifying Inspection", "Bluff"]},
                {"name": "Mob Enforcer 'Knuckles' Malone", "role": "Boss", "stats": {"hp": 320, "speed": "Slow"}, "abilities": ["Lead Pipe Strike", "Tommy Gun Burst"]}
            ],
            "weapons": [
                {"name": ".38 Snubnose Revolver", "type": "Concealed Sidearm", "mechanic": "Emergency defense weapon with scarce 6-round cylinder capacity"}
            ],
            "props": [
                {"name": "Bloodstained Manifest Ledger", "type": "Key Evidence", "mechanic": "Decodes smuggling shipments to reveal killer's identity in notebook"}
            ]
        }
    },
    {
        "input": "An occult bookstore owner researching cursed grimoires in a rain-soaked London townhouse.",
        "target": {
            "game_title": "The Esoteric Codex",
            "genre": "Isometric Narrative Mystery",
            "visual_style": "Muted Antique Victorian Pixel Art",
            "core_loop": {
                "mechanic": "Text translating, ward carving, and sanity stabilization",
                "objective": "Banish the entity summoned by the King in Yellow manuscript",
                "hazards": ["Whispering shadows in corners", "Spontaneous library book fires", "Mind-shattering glyphs"]
            },
            "world_setting": {
                "environment": "Grimoire Antiquities Shop",
                "atmosphere": "Rain beating on stained glass, ticking grandfather clocks, and smell of old parchment",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Archivist Beatrice Wood", "role": "Protagonist", "stats": {"hp": 75, "speed": "Normal"}, "abilities": ["Rune Chant", "Tea Brew Sanity"]},
                {"name": "The Ink-Skin Lurker", "role": "Boss", "stats": {"hp": 300, "speed": "Fast"}, "abilities": ["Page Tear Slash", "Illusion Duplication"]}
            ],
            "weapons": [
                {"name": "Silver Letter Opener", "type": "Occult Athame", "mechanic": "Inscribed with protective glyphs to banish ink-formed phantoms"}
            ],
            "props": [
                {"name": "Heavy Mahogany Reading Desk", "type": "Investigation Station", "mechanic": "Combines fragmented torn codex pages into decipherable rituals"}
            ]
        }
    },
    {
        "input": "A paranormal investigator exploring an abandoned luxury cruise liner drifting in the Bermuda Triangle.",
        "target": {
            "game_title": "S.S. Phantasmagoria",
            "genre": "Top-Down Mystery Exploration",
            "visual_style": "Eerie Faded Art-Deco Pixel Art",
            "core_loop": {
                "mechanic": "EMF meter frequency scanning, spirit voice recording, and deck searching",
                "objective": "Uncover why 2,000 passengers vanished in the 1928 voyage",
                "hazards": ["Phantom ballroom stampedes", "Flooding lower decks", "Temporal time-loop traps"]
            },
            "world_setting": {
                "environment": "Grand Ballroom and Sun Deck",
                "atmosphere": "Faded gold leaf trim, ghostly big-band jazz echoes, and dark ocean void",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Investigator Clara Oswald", "role": "Protagonist", "stats": {"hp": 85, "speed": "Fast"}, "abilities": ["Flash Camera Stun", "Spirit Box Audio"]},
                {"name": "The Drowned Captain", "role": "Boss", "stats": {"hp": 390, "speed": "Normal"}, "abilities": ["Ghost Ship Siren", "Anchor Crash"]}
            ],
            "weapons": [
                {"name": "High-Output UV Flare Gun", "type": "Spectral Disruptor", "mechanic": "Dissolves supernatural apparitions and illuminates hidden wall writing"}
            ],
            "props": [
                {"name": "Ship Engine Telegraph", "type": "Story Trigger", "mechanic": "Aligns the ship chronometer to reset time-loops in the area"}
            ]
        }
    },
    {
        "input": "A rookie police detective trapped inside an eccentric billionaire's puzzle-locked mansion.",
        "target": {
            "game_title": "The Glass Labyrinth",
            "genre": "Escape Room Puzzle Adventure",
            "visual_style": "Modern Crisp Isometric Pixel Art",
            "core_loop": {
                "mechanic": "Mechanical puzzle solving, cipher cracking, and trap disarming",
                "objective": "Unlock the atrium blast gates and escape before the mansion poisons the air",
                "hazards": ["Poisonous neurotoxin vents", "Electrified floor tiles", "Crushing bookshelf walls"]
            },
            "world_setting": {
                "environment": "Kensington Clockwork Estate",
                "atmosphere": "Gleaming marble floors, ticking mechanical walls, and mocking automated speaker voice",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Officer Liam Cross", "role": "Protagonist", "stats": {"hp": 95, "speed": "Fast"}, "abilities": ["Wire Snipping", "Deduction Flash"]},
                {"name": "The Architect's Automaton", "role": "Boss", "stats": {"hp": 350, "speed": "Normal"}, "abilities": ["Gear Throw", "Laser Sight Lock"]}
            ],
            "weapons": [
                {"name": "Heavy Duty Crowbar", "type": "Utility Bludgeon", "mechanic": "Pries open locked ventilation grates and defends against clockwork units"}
            ],
            "props": [
                {"name": "Stained Glass Light Prism", "type": "Laser Puzzle Node", "mechanic": "Rotates light beams to strike target optical receptors on exits"}
            ]
        }
    },
    {
        "input": "An antique restorer discovering murder memories trapped inside Victorian paintings.",
        "target": {
            "game_title": "Canvas of Sins",
            "genre": "Surreal Puzzle Adventure",
            "visual_style": "Oil Painting Textured Pixel Art",
            "core_loop": {
                "mechanic": "Painting portal entering, color pigment restoration, and clue retrieval",
                "objective": "Identify the serial poisoner by repairing the final gallery portrait",
                "hazards": ["Melting pigment vortexes", "Razor-sharp paint scrapers", "Nightmarish canvas tears"]
            },
            "world_setting": {
                "environment": "The Surrealist Gallery",
                "atmosphere": "Dreamlike melting clocks, floating brush strokes, and melancholic cello score",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Restorer Evelyn Vance", "role": "Protagonist", "stats": {"hp": 80, "speed": "Normal"}, "abilities": ["Turpentine Dissolve", "Palette Shield"]},
                {"name": "The Phantom in Crimson", "role": "Boss", "stats": {"hp": 340, "speed": "Fast"}, "abilities": ["Pigment Splatter", "Canvas Slash"]}
            ],
            "weapons": [
                {"name": "Gilded Palette Knife", "type": "Slashing Utility", "mechanic": "Cuts through corrupted canvas layers and parries spectral brush attacks"}
            ],
            "props": [
                {"name": "Antique Painting Easel", "type": "Portal Gateway", "mechanic": "Allows player to dive inside the painted memory to relive past events"}
            ]
        }
    },

    # ----------------- 9. DUNGEON CRAWLER (41-45) -----------------
    {
        "input": "A dwarven berserker descending through ancient subterranean monster halls.",
        "target": {
            "game_title": "Depths of Karak",
            "genre": "Classic 2D Dungeon Crawler",
            "visual_style": "Chunky 16-bit Arcade Pixel Art",
            "core_loop": {
                "mechanic": "Dual-axe cleaving, rage meter building, and door kicking",
                "objective": "Slay the Cave Dragon and reclaim the Crown of the First King",
                "hazards": ["Spike pits with skeletal remains", "Swinging pendulum blade traps", "Magma river falls"]
            },
            "world_setting": {
                "environment": "Forgotten Halls of Karak-Drak",
                "atmosphere": "Torch-lit granite pillars, smell of sulfur, and echoing dwarven war horns",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Borek Ironbreaker", "role": "Protagonist", "stats": {"hp": 160, "speed": "Normal"}, "abilities": ["Berserker Rage", "Earthquake Stomp", "Axe Throw"]},
                {"name": "Ignis the Red Drake", "role": "Boss", "stats": {"hp": 750, "speed": "Fast"}, "abilities": ["Infernal Breath", "Tail Whip Stun"]}
            ],
            "weapons": [
                {"name": "Rune-Carved Battleaxes", "type": "Dual Melee", "mechanic": "Increases attack speed by 5% with each consecutive hit up to 50%"}
            ],
            "props": [
                {"name": "Ancestral Dwarven Anvil", "type": "Forge Shrine", "mechanic": "Repairs broken armor and infuses weapons with elemental fire runes"}
            ]
        }
    },
    {
        "input": "A necromancer summoning skeleton minions to raid a paladin order's crypt.",
        "target": {
            "game_title": "Lord of the Bone Crypt",
            "genre": "Action Strategy Dungeon Crawler",
            "visual_style": "Dark Fantasy 16-bit Pixel Art",
            "core_loop": {
                "mechanic": "Corpse reanimation, minion swarm commanding, and dark spellcasting",
                "objective": "Corrupt the Holy Wellspring in the deepest sanctuary catacomb",
                "hazards": ["Consecrated holy light beams", "Sun-glyph explosion runes", "Silver spike grates"]
            },
            "world_setting": {
                "environment": "Catacombs of the Silver Hand",
                "atmosphere": "Glittering white marble tombs stained with creeping dark purple rot",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Malakor the Defiler", "role": "Protagonist", "stats": {"hp": 85, "speed": "Normal"}, "abilities": ["Raise Skeleton Army", "Bone Armor", "Soul Siphon"]},
                {"name": "Grand Inquisitor Michael", "role": "Boss", "stats": {"hp": 520, "speed": "Fast"}, "abilities": ["Radiant Smite", "Holy Purge Aura"]}
            ],
            "weapons": [
                {"name": "Staff of the Charnel Pit", "type": "Dark Magic Focus", "mechanic": "Fires death bolts that explode corpses into bone shrapnel"}
            ],
            "props": [
                {"name": "Desecrated Sarcophagus", "type": "Minion Spawner", "mechanic": "Supplies fresh skeletal warrior reinforcements when charged with mana"}
            ]
        }
    },
    {
        "input": "A treasure-hunting cat burglar exploring an ever-shifting labyrinth of golden loot.",
        "target": {
            "game_title": "Plunder Maze: Gold Rush",
            "genre": "Isometric Roguelike Dungeon Crawler",
            "visual_style": "Bright Retro 16-bit Gold Pixel Art",
            "core_loop": {
                "mechanic": "Grid-based tactical movement, gold magnet dragging, and trap leaping",
                "objective": "Collect 10,000 gold coins and escape through the teleport ladder",
                "hazards": ["Mimic treasure chests", "Rolling boulder traps", "Gold-siphoning slime puddles"]
            },
            "world_setting": {
                "environment": "The Treasury of Midas",
                "atmosphere": "Piles of gleaming golden goblets and glittering emerald sconces",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Felix the Lucky", "role": "Protagonist", "stats": {"hp": 90, "speed": "Very Fast"}, "abilities": ["Gold Magnet", "Lucky Dodge", "Coin Toss Distract"]},
                {"name": "The Gilded Minotaur", "role": "Boss", "stats": {"hp": 480, "speed": "Fast"}, "abilities": ["Bull Charge", "Ground Tremor"]}
            ],
            "weapons": [
                {"name": "Gilded Sling", "type": "Ranged Ballistic", "mechanic": "Shoots gold coins as ammunition dealing damage proportional to coin value"}
            ],
            "props": [
                {"name": "Golden Midas Fountain", "type": "Greed Altar", "mechanic": "Gambles half your current coins for a chance at legendary artifacts"}
            ]
        }
    },
    {
        "input": "A cursed blood knight sacrificing life points to unleash devastating dark slash attacks.",
        "target": {
            "game_title": "Sanguine Descent",
            "genre": "Gothic 2D Dungeon Crawler",
            "visual_style": "Dark Scarlet High-Contrast Pixel Art",
            "core_loop": {
                "mechanic": "Health-for-power trade-offs, blood siphon parrying, and speed slicing",
                "objective": "Defeat the Crimson Matron to cleanse the hereditary vampiric bloodline",
                "hazards": ["Blood-fountain geysers", "Iron spike iron maidens", "Sanguine acid moats"]
            },
            "world_setting": {
                "environment": "The Sanguine Catacombs",
                "atmosphere": "Rivers of flowing crimson blood illuminated by flickering red candles",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Lord Valerie", "role": "Protagonist", "stats": {"hp": 140, "speed": "Fast"}, "abilities": ["Blood Blade Surge", "Vampiric Drain", "Sanguine Dash"]},
                {"name": "The Crimson Matron", "role": "Boss", "stats": {"hp": 620, "speed": "Very Fast"}, "abilities": ["Blood Puppet Dance", "Exsanguination"]}
            ],
            "weapons": [
                {"name": "The Flayed Greatsword", "type": "Heavy Cursed Blade", "mechanic": "Drains 5 HP per swing but triggers wide crimson cleave waves"}
            ],
            "props": [
                {"name": "Chalice of Sanguine Vigor", "type": "Blood Altar", "mechanic": "Sacrifices maximum HP in exchange for 50% permanent attack boost"}
            ]
        }
    },
    {
        "input": "An arcane archer delving into an overgrown subterranean jungle dungeon.",
        "target": {
            "game_title": "Undergrowth: Sunken Spire",
            "genre": "Top-Down Twin-Stick Dungeon Crawler",
            "visual_style": "Lush Tropical 16-bit Pixel Art",
            "core_loop": {
                "mechanic": "Trick-arrow ricochets, vine swinging, and botanical seed planting",
                "objective": "Cleanse the primal Mother Root in the dungeon center",
                "hazards": ["Spitting carnivorous flower pods", "Quicksand mud pools", "Thorn bramble walls"]
            },
            "world_setting": {
                "environment": "Subterranean Overgrown Canopy",
                "atmosphere": "Dappled light filtering through root ceilings, hum of massive jungle insects",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Ranger Caelen", "role": "Protagonist", "stats": {"hp": 95, "speed": "Fast"}, "abilities": ["Bramble Shot", "Vine Whip Grapple", "Camouflage"]},
                {"name": "Primordial Blossom Wyrm", "role": "Boss", "stats": {"hp": 540, "speed": "Normal"}, "abilities": ["Pollen Cloud Stun", "Root Eruption"]}
            ],
            "weapons": [
                {"name": "Living Oak Longbow", "type": "Magical Archery", "mechanic": "Arrows ricochet off dungeon stone walls up to 3 times before bursting"}
            ],
            "props": [
                {"name": "Primal Bloom Pod", "type": "Flora Hazard/Ally", "mechanic": "Can be shot to release either a healing nectar pool or toxic spore blast"}
            ]
        }
    },

    # ----------------- 10. FARMING / SIMULATION (46-50) -----------------
    {
        "input": "A cozy witch cultivating magical herbs and brewing potions in a secluded forest glade.",
        "target": {
            "game_title": "Witchcrafter: Herb and Cauldron",
            "genre": "Cozy Farming / Potion Sim",
            "visual_style": "Pastel Whimsical 16-bit Pixel Art",
            "core_loop": {
                "mechanic": "Seed sowing, lunar water harvesting, and cauldron recipe mixing",
                "objective": "Fulfill villager potion orders and host the annual Solstice Fair",
                "hazards": ["Weed blight spread", "Mischievous fairy infestations", "Over-boiling cauldron explosions"]
            },
            "world_setting": {
                "environment": "Moonflower Cottage Clearing",
                "atmosphere": "Warm sunset twilight with glowing fireflies and gentle acoustic guitar music",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Willow the Apprentice", "role": "Protagonist", "stats": {"hp": 100, "speed": "Normal"}, "abilities": ["Growth Cantrip", "Broomstick Flight", "Cat Petting"]},
                {"name": "Bramble Imp Prankster", "role": "Enemy", "stats": {"hp": 40, "speed": "Very Fast"}, "abilities": ["Crop Steal", "Acorn Throw"]}
            ],
            "weapons": [
                {"name": "Enchanted Gardening Hoe", "type": "Magical Farm Tool", "mechanic": "Tills magical soil plots in 3x3 grids and banishes thieving pests"}
            ],
            "props": [
                {"name": "Bubbling Cast-Iron Cauldron", "type": "Crafting Station", "mechanic": "Combines harvested botanical ingredients into sellable magical remedies"}
            ]
        }
    },
    {
        "input": "An interstellar colonist growing alien crops inside automated bio-domes on a harsh moon.",
        "target": {
            "game_title": "Astro-Harvest",
            "genre": "Sci-Fi Colony Farming Sim",
            "visual_style": "Crisp Retro-Futuristic Pixel Art",
            "core_loop": {
                "mechanic": "Hydroponic irrigation, nutrient cycling, and climate regulator repairing",
                "objective": "Harvest 500 crates of Star-Melons to feed the orbital colony fleet",
                "hazards": ["Bio-dome radiation leaks", "Micro-meteorite glass punctures", "Alien root parasites"]
            },
            "world_setting": {
                "environment": "Hydro-Dome Beta-4",
                "atmosphere": "Clean white hydroponic bays lit by ultraviolet purple growth lights",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Botanist Leo Vance", "role": "Protagonist", "stats": {"hp": 95, "speed": "Normal"}, "abilities": ["Hydro-Sprayer", "Drone Harvester Recall"]},
                {"name": "Parasitic Mite Queen", "role": "Enemy", "stats": {"hp": 80, "speed": "Fast"}, "abilities": ["Vine Gnaw", "Spore Spray"]}
            ],
            "weapons": [
                {"name": "Laser Weeder Tool", "type": "Agricultural Energy", "mechanic": "Vaporizes invasive space weeds without damaging domestic crop roots"}
            ],
            "props": [
                {"name": "Atmospheric Nutrient Tank", "type": "Life Support", "mechanic": "Pumps nitrogen-rich mist through irrigation lines to speed crop maturity"}
            ]
        }
    },
    {
        "input": "A blacksmith-farmer restoring a war-torn village by forging tools and tending livestock.",
        "target": {
            "game_title": "Anvil and Acres",
            "genre": "Cozy Medieval Farming and Crafting",
            "visual_style": "Warm Earthy 16-bit Pixel Art",
            "core_loop": {
                "mechanic": "Plow tilling, anvil hammer timing, and cow milking",
                "objective": "Rebuild the ruined village bridge and host the Autumn Harvest Feast",
                "hazards": ["Nighttime wolf raids", "Crop mildew rainstorms", "Broken furnace overheating"]
            },
            "world_setting": {
                "environment": "Riverbend Valley Farmstead",
                "atmosphere": "Golden morning sunlight, rooster crowing, and fragrant smell of baked bread",
                "camera_view": "Top-Down 2D"
            },
            "characters": [
                {"name": "Garrick the Smith", "role": "Protagonist", "stats": {"hp": 120, "speed": "Normal"}, "abilities": ["Forge Hammer Strike", "Hay Bale Toss"]},
                {"name": "Shadow Taiga Wolf", "role": "Enemy", "stats": {"hp": 60, "speed": "Fast"}, "abilities": ["Sheep Lunge", "Threatening Growl"]}
            ],
            "weapons": [
                {"name": "Heavy Iron Pitchfork", "type": "Agricultural Polearm", "mechanic": "Gathers hay bales and repels predators with sweeping wide arcs"}
            ],
            "props": [
                {"name": "Village Blast Forge", "type": "Crafting Station", "mechanic": "Smelts dug iron ore into upgraded farm tools and horseshoe horseshoes"}
            ]
        }
    },
    {
        "input": "An underwater mermaid farmer domesticating luminescent kelp and sea turtles.",
        "target": {
            "game_title": "Coral Grove",
            "genre": "Aquatic Farming Adventure",
            "visual_style": "Vibrant Aquatic Bioluminescent Pixel Art",
            "core_loop": {
                "mechanic": "Current riding, pearl diving, and sea anemone tending",
                "objective": "Restore the Great Reef Mother Coral from oceanic bleaching",
                "hazards": ["Thermal volcanic vents", "Venomous jellyfish drifts", "Predatory moray eels"]
            },
            "world_setting": {
                "environment": "The Sunken Atoll",
                "atmosphere": "Turquoise waters rippling with sun caustics and schools of neon clownfish",
                "camera_view": "Side-Scrolling 2D"
            },
            "characters": [
                {"name": "Coralia the Reefkeeper", "role": "Protagonist", "stats": {"hp": 100, "speed": "Fast"}, "abilities": ["Bubble Net", "Turtle Ride", "Current Boost"]},
                {"name": "Giant Moray Eel", "role": "Enemy", "stats": {"hp": 90, "speed": "Very Fast"}, "abilities": ["Lunge Ambush", "Tail Whirlpool"]}
            ],
            "weapons": [
                {"name": "Mother-of-Pearl Trident", "type": "Underwater Polearm", "mechanic": "Directs water currents and repels territorial reef predators"}
            ],
            "props": [
                {"name": "Bioluminescent Clam Bed", "type": "Pearl Incubator", "mechanic": "Produces valuable luminous pearls over a 3-day ocean tide cycle"}
            ]
        }
    },
    {
        "input": "A post-apocalyptic botanist cultivating hardy rooftop greenhouse crops above a flooded city.",
        "target": {
            "game_title": "Sky-Sprout: Flooded City",
            "genre": "Survival Farming / Management",
            "visual_style": "Lush Overgrown Post-Apocalyptic 32-bit",
            "core_loop": {
                "mechanic": "Rainwater distillation, zip-line crop transport, and seagull scaring",
                "objective": "Build a sustainable aerial botanical network across three skyscraper roofs",
                "hazards": ["Acid rain storms", "Rusted rooftop solar panel collapse", "Scavenger crow swarms"]
            },
            "world_setting": {
                "environment": "Skyscraper Canopy Complex",
                "atmosphere": "Overcast humid skies, ocean waves lapping at lower skyscraper floors",
                "camera_view": "Isometric"
            },
            "characters": [
                {"name": "Maya the Arborist", "role": "Protagonist", "stats": {"hp": 100, "speed": "Normal"}, "abilities": ["Zip-Line Slide", "Seed Bomb Lob", "Rain Collector Check"]},
                {"name": "Mutant Gilded Crow", "role": "Enemy", "stats": {"hp": 45, "speed": "Very Fast"}, "abilities": ["Fruit Dive", "Claw Scratch"]}
            ],
            "weapons": [
                {"name": "Modified Air-Powered Seed Rifle", "type": "Pneumatic Planter", "mechanic": "Shoots seeds into distant soil beds and stuns crop-stealing pests"}
            ],
            "props": [
                {"name": "Rooftop Rainwater Cistern", "type": "Water Reservoir", "mechanic": "Collects fresh rainwater during storms to irrigate greenhouse crops"}
            ]
        }
    }
]

assert len(examples) == 50, f"Expected 50 examples, got {len(examples)}"

# Validate every single target JSON string and schema
for idx, ex in enumerate(examples, 1):
    assert "input" in ex, f"Missing 'input' in example {idx}"
    assert "target" in ex, f"Missing 'target' in example {idx}"
    t = ex["target"]
    required_keys = ["game_title", "genre", "visual_style", "core_loop", "world_setting", "characters", "weapons", "props"]
    for k in required_keys:
        assert k in t, f"Missing '{k}' in target of example {idx}"
    assert len(t["characters"]) == 2, f"Example {idx} must have 2 characters"
    assert len(t["weapons"]) >= 1, f"Example {idx} must have >= 1 weapon"
    assert len(t["props"]) >= 1, f"Example {idx} must have >= 1 prop"

# Prepare JSONL output
jsonl_lines = []
for ex in examples:
    # Target must be escaped JSON string as required: {"input": "...", "target": "{\"...\"}"}
    target_str = json.dumps(ex["target"], separators=(',', ':'))
    record = {
        "input": ex["input"],
        "target": target_str
    }
    jsonl_lines.append(json.dumps(record, separators=(',', ':')))

# Write to file
output_path = os.path.join("submission", "flan_t5_game_design_50_dataset.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for line in jsonl_lines:
        f.write(line + "\n")

print(f"✅ Successfully created and validated 50 JSONL records at: {output_path}")
