import json

src_path = "Notebook/training/03_DCGAN_Centroid_FeatureMatching_Pilot_10Epochs.ipynb"
dst_path = "Notebook/training/02_DCGAN_FrontFacing_46K_50Epochs.ipynb"

with open(src_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 0 header
nb["cells"][0]["source"] = [
    "# 🎮 GameForge AI: 46K Audited Front-Facing DCGAN Training (50 Epochs)\n",
    "\n",
    "**Experiment:** Full 50-Epoch Unconditional DCGAN Character Generation Baseline  \n",
    "**Dataset:** 42,212 Training Sprites (`final_alucard_frontfacing_manifest.csv`)  \n",
    "**Target Architecture:** Bit-Exact Replica of `backend/dcgan_service.py` (5,402,181 Total Params)  \n"
]

# Update Cell 3 config: EPOCHS = 50, WORK_DIR = dcgan_frontfacing_46k_50ep_results
c3_src = "".join(nb["cells"][3]["source"])
c3_src = c3_src.replace("EPOCHS                 = 10", "EPOCHS                 = 50")
c3_src = c3_src.replace("dcgan_frontfacing_centroid_pilot_results", "dcgan_frontfacing_46k_50ep_results")
nb["cells"][3]["source"] = [line + "\n" for line in c3_src.split("\n") if line]

# Update Cell 7 to original unpaired FM for historical fidelity of the 50-ep experiment
c7_src = "".join(nb["cells"][7]["source"])
c7_src = c7_src.replace(
    """        # CANONICAL BATCH CENTROID FEATURE MATCHING (Salimans et al., 2016)
        # Average feature representations across batch dimension (axis 0) before distance:
        mean_real = tf.stop_gradient(tf.reduce_mean(real_feats, axis=0))
        mean_fake = tf.reduce_mean(fake_feats, axis=0)
        fm_loss = tf.reduce_mean(tf.abs(mean_real - mean_fake)) / float(strategy.num_replicas_in_sync)""",
    """        # Feature Matching Loss (Notebook baseline unpaired elementwise formulation)
        fm_loss = tf.reduce_mean(tf.abs(real_feats - fake_feats)) / float(strategy.num_replicas_in_sync)"""
)
nb["cells"][7]["source"] = [line + "\n" for line in c7_src.split("\n") if line]

# Update Cell 11 training loop epochs: periodic checkpoints at 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
c11_src = "".join(nb["cells"][11]["source"])
c11_src = c11_src.replace("if epoch in [1, 5, 10]:", "if epoch in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:")
nb["cells"][11]["source"] = [line + "\n" for line in c11_src.split("\n") if line]

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully rebuilt {dst_path} as valid JSON notebook with {len(nb['cells'])} cells.")

