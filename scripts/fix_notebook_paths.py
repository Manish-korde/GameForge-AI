import json

nb_path = r'c:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator\Notebook\280K_VAE_Evaluation.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "encoder_path =" in source:
            new_source = []
            for line in cell['source']:
                if "encoder_path =" in line:
                    new_source.append("encoder_path = os.path.join(PROJECT_ROOT, 'models', '280k model VAE (VAE v2)', 'VAE_280K_Outputs', 'encoder_280k_final.keras')\n")
                elif "decoder_path =" in line:
                    new_source.append("decoder_path = os.path.join(PROJECT_ROOT, 'models', '280k model VAE (VAE v2)', 'VAE_280K_Outputs', 'decoder_280k_final.keras')\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Fixed notebook paths")
