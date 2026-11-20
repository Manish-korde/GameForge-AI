"""
Append full training pipeline cells to Notebook/training/02_DCGAN_FrontFacing_46K_50Epochs.ipynb
"""

import json

nb_path = 'Notebook/training/02_DCGAN_FrontFacing_46K_50Epochs.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 9: Distributed Dataset Pipeline
cell_9_code = '''# Build Distributed Dataset Pipeline from Approved Manifest
from datasets import load_dataset

print("Streaming 'evilsocket/alucard-sprites' from HuggingFace Hub...")
raw_ds = load_dataset("evilsocket/alucard-sprites", split="train")
print(f"Loaded raw dataset stream: {len(raw_ds):,} entries.")

# Extract training array using archive_index from train_df
train_indices = train_df['archive_index'].values
print(f"Pre-caching {len(train_indices):,} training sprites into memory...")

def load_and_preprocess_dataset(indices_list):
    tensors = np.zeros((len(indices_list), IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
    t0 = time.time()
    for idx, arch_i in enumerate(indices_list):
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx:,}/{len(indices_list):,} sprites ({time.time()-t0:.1f}s)...")
        img = raw_ds[int(arch_i)]["image"].convert("RGBA")
        if img.size != (IMG_WIDTH, IMG_HEIGHT):
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.float32)
        # Standard normalization [0, 255] -> [-1.0, 1.0]
        tensors[idx] = (arr / 127.5) - 1.0
    print(f"Completed tensor preparation in {time.time()-t0:.1f}s. Shape: {tensors.shape}")
    return tensors

train_tensors = load_and_preprocess_dataset(train_indices)

# Build tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices(train_tensors)
train_ds = train_ds.shuffle(buffer_size=min(10000, len(train_tensors)), seed=SEED)
train_ds = train_ds.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# Distribute dataset across GPU replicas
dist_train_ds = strategy.experimental_distribute_dataset(train_ds)

STEPS_PER_EPOCH = len(train_tensors) // GLOBAL_BATCH_SIZE
print(f"Distributed Dataset Ready: {STEPS_PER_EPOCH:,} steps per epoch (Batch Size: {GLOBAL_BATCH_SIZE})")
print(f"Total Gradient Updates across {EPOCHS} Epochs: {STEPS_PER_EPOCH * EPOCHS:,}")'''

# Cell 10: Full 50-Epoch Training Loop with Periodic Checkpointing & Fixed Latent Grids
cell_10_code = '''# 50-Epoch Full Distributed Training Loop
CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
history = {"epoch": [], "d_loss": [], "g_loss": [], "adv_loss": [], "fm_loss": [], "duration_sec": []}
history_csv_path = f"{LOG_DIR}/training_history.csv"

def save_sample_grid(epoch_num):
    """Generates sample grid using fixed latent seeds for longitudinal comparison."""
    eval_preds = generator(fixed_latents[:36], training=False).numpy()
    eval_sprites = np.clip((eval_preds + 1.0) * 127.5, 0, 255).astype(np.uint8)
    
    fig, axes = plt.subplots(6, 6, figsize=(10, 10), facecolor="#0B0F19")
    for ax, sprite in zip(axes.flatten(), eval_sprites):
        ax.imshow(sprite)
        ax.axis("off")
    plt.suptitle(f"DCGAN 46K Front-Facing — Epoch {epoch_num:02d} Fixed Latents", color="#F9FAFB", fontsize=14, y=0.92)
    grid_path = f"{SAMPLE_DIR}/epoch_{epoch_num:02d}_fixed_grid.png"
    plt.savefig(grid_path, dpi=200, bbox_inches="tight", facecolor="#0B0F19")
    plt.close()

print(f"STARTING 50-EPOCH TRAINING RUN (Global Batch Size: {GLOBAL_BATCH_SIZE})")
print("=" * 80)
total_train_t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    ep_t0 = time.time()
    ep_d_loss, ep_g_loss, ep_adv_loss, ep_fm_loss = 0.0, 0.0, 0.0, 0.0
    step_count = 0
    
    for batch in dist_train_ds:
        d_l, g_l, adv_l, fm_l = distributed_train_step(batch)
        ep_d_loss += float(d_l)
        ep_g_loss += float(g_l)
        ep_adv_loss += float(adv_l)
        ep_fm_loss += float(fm_l)
        step_count += 1
        
    ep_duration = time.time() - ep_t0
    mean_d = ep_d_loss / step_count
    mean_g = ep_g_loss / step_count
    mean_adv = ep_adv_loss / step_count
    mean_fm = ep_fm_loss / step_count
    
    history["epoch"].append(epoch)
    history["d_loss"].append(mean_d)
    history["g_loss"].append(mean_g)
    history["adv_loss"].append(mean_adv)
    history["fm_loss"].append(mean_fm)
    history["duration_sec"].append(ep_duration)
    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({ep_duration:5.1f}s) | "
          f"D_Loss: {mean_d:.4f} | G_Loss: {mean_g:.4f} (Adv: {mean_adv:.4f}, FM: {mean_fm:.4f})")
          
    if epoch in CHECKPOINT_EPOCHS:
        save_sample_grid(epoch)
        generator.save_weights(f"{CKPT_DIR}/generator_epoch_{epoch:02d}.weights.h5")
        discriminator.save_weights(f"{CKPT_DIR}/discriminator_epoch_{epoch:02d}.weights.h5")
        print(f"  -> Checkpoint & Fixed Grid saved for Epoch {epoch:02d}")

# Save Final Weights
generator.save_weights(f"{WORK_DIR}/generator_weights_final.weights.h5")
discriminator.save_weights(f"{WORK_DIR}/discriminator_weights_final.weights.h5")
total_duration = time.time() - total_train_t0
print("=" * 80)
print(f"50-EPOCH TRAINING COMPLETE in {total_duration/60:.1f} minutes!")
print(f"Final Weights Saved to: {WORK_DIR}/generator_weights_final.weights.h5")'''

# Cell 11: Master 100-Character Showcase & Evaluation
cell_11_code = '''# Generate Master 100-Character Showcase & Final Loss Curves
print("Generating Final 100-Character Master Showcase...")
eval_preds = generator(fixed_latents, training=False).numpy()
eval_sprites = np.clip((eval_preds + 1.0) * 127.5, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(10, 10, figsize=(18, 18), facecolor="#0B0F19")
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for ax, sprite in zip(axes.flatten(), eval_sprites):
    ax.imshow(sprite)
    ax.axis("off")
plt.suptitle("GameForge AI: DCGAN 46K Audited Front-Facing Master Showcase (100 Characters)", color="#F9FAFB", fontsize=16, y=0.91)
showcase_path = f"{WORK_DIR}/final_master_showcase_100_characters.png"
plt.savefig(showcase_path, dpi=250, bbox_inches="tight", facecolor="#0B0F19")
plt.close()
print(f"Saved master showcase to: {showcase_path}")

# Plot Loss Curves
hist_df = pd.DataFrame(history)
fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="#0B0F19")
ax1.set_facecolor("#111827")
ax1.plot(hist_df["epoch"], hist_df["d_loss"], label="D Loss (LSGAN)", color="#EF4444", lw=2)
ax1.plot(hist_df["epoch"], hist_df["g_loss"], label="G Loss (Total)", color="#3B82F6", lw=2)
ax1.plot(hist_df["epoch"], hist_df["adv_loss"], label="G Adversarial", color="#10B981", lw=1.5, ls="--")
ax1.plot(hist_df["epoch"], hist_df["fm_loss"], label="Feature Matching", color="#F59E0B", lw=1.5, ls=":")
ax1.set_xlabel("Epoch", color="#F9FAFB")
ax1.set_ylabel("Loss", color="#F9FAFB")
ax1.tick_params(colors="#9CA3AF")
ax1.legend(loc="upper right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB")
ax1.set_title("DCGAN 46K Front-Facing Training Dynamics (50 Epochs)", color="#F9FAFB", fontsize=12)
ax1.grid(True, alpha=0.2)
curve_path = f"{LOG_DIR}/loss_curves.png"
plt.savefig(curve_path, dpi=200, bbox_inches="tight", facecolor="#0B0F19")
plt.close()
print(f"Saved loss curves to: {curve_path}")'''

if len(nb['cells']) == 9:
    nb['cells'].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_9_code.split('\n')]})
    nb['cells'].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_10_code.split('\n')]})
    nb['cells'].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in cell_11_code.split('\n')]})

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Appended training cells 9, 10, 11 to {nb_path}. Total cells now: {len(nb['cells'])}")

