import json

with open(r'c:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator\Notebook\280K_Autoencoder_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
        
    source_text = "".join(cell.get('source', []))
    
    # 1. Environment
    if "EVAL_DIR = os.path.join(PROJECT_ROOT, 'evaluation')" in source_text:
        for i, line in enumerate(cell['source']):
            if "EVAL_DIR =" in line:
                cell['source'][i] = "EVAL_DIR = os.path.join(PROJECT_ROOT, 'evaluation_vae')\n"
    
    # 2. Load Model
    if "autoencoder = tf.keras.models.load_model" in source_text:
        cell['source'] = [
            "encoder_path = os.path.join(MODELS_DIR, 'encoder_280k_final.keras')\n",
            "decoder_path = os.path.join(MODELS_DIR, 'decoder_280k_final.keras')\n",
            "print(f\"Loading {encoder_path} and {decoder_path}\")\n",
            "\n",
            "@tf.keras.utils.register_keras_serializable()\n",
            "class Sampling(tf.keras.layers.Layer):\n",
            "    def call(self, inputs):\n",
            "        z_mean, z_log_var = inputs\n",
            "        batch = tf.shape(z_mean)[0]\n",
            "        dim = tf.shape(z_mean)[1]\n",
            "        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))\n",
            "        return z_mean + tf.exp(0.5 * z_log_var) * epsilon\n",
            "\n",
            "layers_to_patch = [\n",
            "    tf.keras.layers.Dense,\n",
            "    tf.keras.layers.Conv2D,\n",
            "    tf.keras.layers.Conv2DTranspose,\n",
            "    tf.keras.layers.Flatten,\n",
            "    tf.keras.layers.Reshape,\n",
            "    tf.keras.layers.InputLayer\n",
            "]\n",
            "for layer_cls in layers_to_patch:\n",
            "    original_init = layer_cls.__init__\n",
            "    def make_patched_init(orig_init):\n",
            "        def patched_init(self, *args, **kwargs):\n",
            "            kwargs.pop('quantization_config', None)\n",
            "            orig_init(self, *args, **kwargs)\n",
            "        return patched_init\n",
            "    layer_cls.__init__ = make_patched_init(original_init)\n",
            "\n",
            "vae_encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})\n",
            "vae_decoder = tf.keras.models.load_model(decoder_path)\n",
            "vae_encoder.summary()\n",
            "vae_decoder.summary()\n"
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        
    # 3. Evaluate
    if "autoencoder.predict_on_batch" in source_text:
        new_source = []
        for line in cell['source']:
            if "preds = autoencoder.predict_on_batch(batch_x)" in line:
                new_source.append("    z_mean, z_log_var, _ = vae_encoder.predict_on_batch(batch_x)\n")
                new_source.append("    preds = vae_decoder.predict_on_batch(z_mean)\n")
            elif "quality_preds = autoencoder.predict(quality_imgs)" in line:
                new_source.append("quality_z_mean, _, _ = vae_encoder.predict(quality_imgs)\n")
                new_source.append("quality_preds = vae_decoder.predict(quality_z_mean)\n")
            elif "280K AUTOENCODER FINAL EVALUATION" in line:
                new_source.append(line.replace("AUTOENCODER", "VAE"))
            elif "Best epoch:" in line or "Best validation MSE:" in line:
                pass
            else:
                new_source.append(line)
        cell['source'] = new_source
        cell['outputs'] = []
        cell['execution_count'] = None

with open(r'c:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator\Notebook\280K_VAE_Evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Saved 280K_VAE_Evaluation.ipynb")
