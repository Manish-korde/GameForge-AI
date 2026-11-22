import json
with open('models/latent_gallery_manifest_25k.json') as f:
    m = json.load(f)

weapons = [x for x in m if x['category'] == 'Weapons' and 'sword' in x['text'].lower()]
items = [x for x in m if x['category'] == 'Items' and 'potion' in x['text'].lower()]
enemies = [x for x in m if x['category'] == 'Enemies']

print(f"Total weapons: {len([x for x in m if x['category'] == 'Weapons'])}")
print(f"Total items: {len([x for x in m if x['category'] == 'Items'])}")
print(f"Total enemies: {len(enemies)}")
print(f"Total characters: {len([x for x in m if x['category'] == 'Characters'])}")

print("\nSample Swords:")
for w in weapons[:5]:
    print(w)

print("\nSample Potions:")
for it in items[:5]:
    print(it)
