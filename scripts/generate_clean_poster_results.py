import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

brain_dir = r"C:\Users\manis\.gemini\antigravity\brain\0e75ca3f-2ae4-4b95-8ef8-c8e5292ad012"
os.makedirs("reports/poster_figures", exist_ok=True)

# Helper function to place an image on pure white background
def paste_on_white(img):
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, (0, 0), img)
        return bg.convert("RGB")
    return img.convert("RGB")

print("=" * 60)
print("1. Creating Image 1: Input Dataset Sheet (White Background)")
print("=" * 60)
# Use the canonical 2400x2400 character sheet from research_paper_assets
raw_ds = Image.open("research_paper_assets/01_dataset_analysis/character.png")
white_ds = paste_on_white(raw_ds)

# Add a clean minimal title bar
ds_canvas = Image.new("RGB", (white_ds.width, white_ds.height + 120), (255, 255, 255))
d_draw = ImageDraw.Draw(ds_canvas)
d_draw.text((60, 40), "Alucard Ground-Truth Sprite Dataset (Curated 64x64 RGBA Characters)", fill=(20, 20, 20))
d_draw.text((60, 80), "Source: evilsocket/alucard-sprites | 282,511 Total Assets | 46,000 Front-Facing Characters", fill=(100, 100, 100))
ds_canvas.paste(white_ds, (0, 120))

out_ds_path = "reports/poster_figures/1_dataset_sheet_white_bg.png"
ds_canvas.save(out_ds_path, quality=95)
ds_canvas.save(os.path.join(brain_dir, "1_dataset_sheet_white_bg.png"), quality=95)
print("[Saved]", out_ds_path)

print("\n" + "=" * 60)
print("2. Creating Image 2: 3-Character Input vs AE vs VAE Comparison (White BG)")
print("=" * 60)
# Both AE and VAE orig vs recon are 600x1500 (2 columns x 5 rows = 10 sprites of 300x300)
# In ae_original_vs_reconstruction.png: Left col = Original, Right col = AE Recon
# In vae_original_vs_reconstruction.png: Left col = Original, Right col = VAE Recon
im_ae = Image.open("research_paper_assets/02_autoencoder_qa_outliers/ae_original_vs_reconstruction.png")
im_vae = Image.open("research_paper_assets/03_vae_latent_morphing/vae_original_vs_reconstruction.png")

# Each cell is roughly 300 wide by 300 high
cell_w, cell_h = 300, 300

comp_canvas = Image.new("RGB", (3 * cell_w + 120, 3 * cell_h + 160), (255, 255, 255))
c_draw = ImageDraw.Draw(comp_canvas)

# Title
c_draw.text((40, 25), "Reconstruction Fidelity Benchmark: Ground Truth vs. Conv AE vs. Beta-VAE", fill=(10, 10, 10))
c_draw.text((40, 55), "Evaluated on 14,125 Held-Out Test Sprites | AE: 35.36 dB PSNR (Sharp Edge QA) | VAE: 26.59 dB PSNR (Smooth Latent Prior)", fill=(80, 80, 80))

# Column Headers
c_draw.text((60 + 0 * (cell_w + 20) + 70, 95), "1. Ground Truth (Input)", fill=(20, 100, 30))
c_draw.text((60 + 1 * (cell_w + 20) + 70, 95), "2. Conv AE (Reconstruction)", fill=(20, 60, 150))
c_draw.text((60 + 2 * (cell_w + 20) + 70, 95), "3. Beta-VAE (Reconstruction)", fill=(180, 60, 20))

# Extract 3 characters (row 0, row 1, row 2)
for row in range(3):
    y_src = row * cell_h
    y_dst = 135 + row * (cell_h + 10)
    
    # Original sprite from AE left col
    orig_sprite = im_ae.crop((0, y_src, cell_w, y_src + cell_h))
    # AE recon from AE right col
    ae_sprite = im_ae.crop((cell_w, y_src, 2 * cell_w, y_src + cell_h))
    # VAE recon from VAE right col
    vae_sprite = im_vae.crop((cell_w, y_src, 2 * cell_w, y_src + cell_h))
    
    comp_canvas.paste(orig_sprite, (60 + 0 * (cell_w + 20), y_dst))
    comp_canvas.paste(ae_sprite, (60 + 1 * (cell_w + 20), y_dst))
    comp_canvas.paste(vae_sprite, (60 + 2 * (cell_w + 20), y_dst))

out_comp_path = "reports/poster_figures/2_input_vs_ae_vs_vae_comparison.png"
comp_canvas.save(out_comp_path, quality=95)
comp_canvas.save(os.path.join(brain_dir, "2_input_vs_ae_vs_vae_comparison.png"), quality=95)
print("[Saved]", out_comp_path)

print("\n" + "=" * 60)
print("3. Creating Image 3: DCGAN Sheet Result (White Background)")
print("=" * 60)
# We load the final discrete DCGAN epoch 50 samples sheet (1500x1500)
raw_dcgan = Image.open("models/3_DCGAN_42k_v1/samples/samples_epoch_0050.png")
white_dcgan = paste_on_white(raw_dcgan)

dcgan_canvas = Image.new("RGB", (white_dcgan.width, white_dcgan.height + 120), (255, 255, 255))
dc_draw = ImageDraw.Draw(dcgan_canvas)
dc_draw.text((50, 35), "Discrete Codebook DCGAN - 64x64 Character Synthesis Sheet (Epoch 50 Locked Weights)", fill=(20, 20, 20))
dc_draw.text((50, 75), "Spatial Coherence: 48.8% (Real: 53.0%) | Unique Colors: 25.7 (Target 32) | Binary Alpha {0, 255} | LCC Cleaned", fill=(90, 90, 90))
dcgan_canvas.paste(white_dcgan, (0, 120))

out_dcgan_path = "reports/poster_figures/3_dcgan_result_sheet_white_bg.png"
dcgan_canvas.save(out_dcgan_path, quality=95)
dcgan_canvas.save(os.path.join(brain_dir, "3_dcgan_result_sheet_white_bg.png"), quality=95)
print("[Saved]", out_dcgan_path)

print("\n" + "=" * 60)
print("4. Creating Image 4: Velocity Diffusion Good Result 8-Frame Walk Cycle (White BG)")
print("=" * 60)
# Use walk_cycle_demo.png from Phase G or walk_contact_sheet from Refined C2
# walk_cycle_demo.png is 664x100 and shows the verified refined kinematic walk
raw_walk = Image.open("research_paper_assets/06_sprite_deformation_and_pose_animation/13_phase_G_diffusion_integration/walk_cycle_demo.png")
white_walk = paste_on_white(raw_walk)

walk_canvas = Image.new("RGB", (white_walk.width + 80, white_walk.height + 110), (255, 255, 255))
w_draw = ImageDraw.Draw(walk_canvas)
w_draw.text((40, 25), "Refined Velocity Diffusion (v-Prediction) - Verified Kinematic 8-Frame Walk Cycle", fill=(15, 15, 15))
w_draw.text((40, 55), "Orthogonal v-Prediction + Spatially Weighted Loss | Correct Leg Articulation | 0.956 Pose IoU | Zero Identity Drift", fill=(80, 80, 80))
walk_canvas.paste(white_walk, (40, 90))

out_walk_path = "reports/poster_figures/4_velocity_diffusion_8frame_walk_white_bg.png"
walk_canvas.save(out_walk_path, quality=95)
walk_canvas.save(os.path.join(brain_dir, "4_velocity_diffusion_8frame_walk_white_bg.png"), quality=95)
print("[Saved]", out_walk_path)

print("\n" + "=" * 60)
print("ALL 4 REQUESTED RESULT IMAGES SUCCESSFULLY GENERATED ON PURE WHITE BACKGROUND!")
print("=" * 60)

