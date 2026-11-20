import json, csv, os

print("=== VERIFYING REAL DISK FILES AND METRICS ===")

# 1. Diffusion CSV
diff_csv = "research_paper_assets/06_sprite_deformation_and_pose_animation/diffusion_quality_fix_150ep_loss_history.csv"
if os.path.exists(diff_csv):
    with open(diff_csv, 'r') as f:
        rows = list(csv.DictReader(f))
        r50 = next(r for r in rows if r['epoch'] == '50')
        r49 = next(r for r in rows if r['epoch'] == '49')
        print(f"[EXISTS] Diffusion CSV Line 50 -> Epoch: {r50['epoch']} | Val Loss: {float(r50['val_loss']):.4f} | Epoch 49 best: {float(r49['val_loss']):.4f}")

# 2. Deformation Baseline JSON
deform_json = "temp files/gameforge_50ep_deformation_artifacts/model_config.json"
if os.path.exists(deform_json):
    with open(deform_json, 'r') as f:
        cfg = json.load(f)
        print(f"[EXISTS] Deformation Baseline JSON -> Model: {cfg['model_name']} | Epochs: {cfg['epochs']} | Val Loss: {cfg['best_val_loss']:.4f} | Pairs: {cfg['training_pairs']}")

# 3. Model weights on disk
models = {
    "Diffusion UNet (best.pt)": "zDIffusion_deformation final stuff/Quality fix new latest/checkpoints/best.pt",
    "Deformation Baseline (.pt)": "temp files/gameforge_50ep_deformation_artifacts/gameforge_deformation_50ep_weights.pt",
    "Autoencoder (.keras)": "models/280k dataset model/AE_280K_best.keras",
    "VAE Encoder (.keras)": "models/280k model VAE (VAE v2)/VAE_280K_Outputs/encoder_280k_final.keras",
    "DCGAN 50ep (.h5)": "models/dcgan/dcgan_169k_50ep.h5"
}

for name, path in models.items():
    exists = os.path.exists(path)
    sz = os.path.getsize(path) / (1024*1024) if exists else 0
    print(f"[EXISTS: {exists}] {name} -> Size: {sz:.2f} MB at: {path}")
