import json

nb_path = "Notebook/training/03_Discrete_SAGAN_Production.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell7_code = '''# Cell 7: Production Training Loop with G-EMA, Gumbel Annealing, Sample Grid Export & Gate Checkpointing
fixed_eval_z = tf.random.normal([36, LATENT_DIM], seed=42)
steps_per_epoch = N_SPRITES // GLOBAL_BATCH_SIZE

history = []
best_gate_score = -1.0
t_start = time.time()

print("Starting Production Discrete SAGAN Training...")
print(f"Total Epochs: {TARGET_EPOCHS} | Steps per Epoch: {steps_per_epoch}")

for epoch in range(1, TARGET_EPOCHS + 1):
    epoch_t0 = time.time()
    
    # Anneal Gumbel Temperature from TAU_START (1.0) to TAU_END (0.5)
    tau_current = TAU_START - (TAU_START - TAU_END) * (float(epoch - 1) / float(max(1, TARGET_EPOCHS - 1)))
    
    d_loss_accum = 0.0
    g_loss_accum = 0.0
    
    for step, real_batch in enumerate(dist_dataset):
        d_l, g_l = distributed_train_step(real_batch, tf.constant(tau_current, dtype=tf.float32))
        d_loss_accum += float(d_l)
        g_loss_accum += float(g_l)
        update_ema()
        
    d_loss_avg = d_loss_accum / steps_per_epoch
    g_loss_avg = g_loss_accum / steps_per_epoch
    
    # Visual Evaluation using EMA Generator (Sample Grid)
    eval_logits = ema_generator(fixed_eval_z, training=False)
    eval_imgs_tf = logits_to_image(eval_logits, tau=tau_current, hard=True)
    eval_imgs = tf.cast((eval_imgs_tf + 1.0) * 127.5, tf.uint8).numpy()
    
    # Save visual sample grid every 2 epochs & epoch 1
    if epoch == 1 or epoch % 2 == 0 or epoch == TARGET_EPOCHS:
        grid = Image.new("RGBA", (6 * 64, 6 * 64), (255, 255, 255, 255))
        for idx in range(36):
            r, c = idx // 6, idx % 6
            p_img = Image.fromarray(eval_imgs[idx])
            grid.paste(p_img, (c * 64, r * 64), mask=p_img.split()[3])
        grid.save(os.path.join(SAMPLES_DIR, f"samples_epoch_{epoch:04d}.png"))
    
    # Compute Visual Gate Metrics
    colors_list = []
    fg_area_list = []
    for img in eval_imgs:
        fg = img[..., 3] > 127
        fg_area_list.append(np.mean(fg))
        if np.any(fg):
            colors_list.append(len(np.unique(img[..., :3][fg], axis=0)))
            
    mean_colors = float(np.mean(colors_list)) if colors_list else 0.0
    mean_fg = float(np.mean(fg_area_list))
    
    # Pairwise Diversity
    flat = eval_imgs.astype(np.float32).reshape((36, -1)) / 255.0
    dists = [np.mean(np.abs(flat[i] - flat[j])) for i in range(36) for j in range(i+1, 36)]
    mean_div = float(np.mean(dists))
    
    # Coherence proxy
    mean_coherence = float(np.clip(1.0 - abs(mean_fg - 0.35) * 2.0, 0.0, 1.0))
    
    # Composite Visual Gate Score
    gate_score = mean_div * mean_coherence * min(1.0, mean_colors / 18.0)
    
    epoch_dur = time.time() - epoch_t0
    print(f"Epoch {epoch:02d}/{TARGET_EPOCHS} ({tau_current:.2f}tau) | D: {d_loss_avg:.4f} | G: {g_loss_avg:.4f} | "
          f"Colors: {mean_colors:.1f} | Div: {mean_div:.3f} | GateScore: {gate_score:.4f} ({epoch_dur:.1f}s)")
          
    history.append({
        "epoch": epoch, "d_loss": d_loss_avg, "g_loss": g_loss_avg,
        "colors": mean_colors, "diversity": mean_div, "gate_score": gate_score
    })
    
    # Save Best Checkpoint by Visual Gate Score (Guarantees we keep the visual peak)
    if gate_score > best_gate_score:
        best_gate_score = gate_score
        ema_generator.save_weights(os.path.join(CHECKPOINTS_DIR, "generator_best_gate.weights.h5"))
        print(f"  --> Saved new best checkpoint (Gate Score: {best_gate_score:.4f})")
        
    # Smoke Gate Abort Check at Epoch 6
    if epoch == SMOKE_GATE_EPOCH:
        if mean_div < 0.04 or mean_colors < 8.0:
            raise RuntimeError(f"ABORTED at Epoch {SMOKE_GATE_EPOCH}: Visual gate failed (Div: {mean_div:.3f}, Colors: {mean_colors:.1f})")
        else:
            print(f"PASS: Epoch {SMOKE_GATE_EPOCH} Smoke Gate verified. Continuing run.")

# Save training history and final weights
with open(os.path.join(LOGS_DIR, "training_history.csv"), "w") as f:
    f.write("epoch,d_loss,g_loss,colors,diversity,gate_score\\n")
    for h in history:
        f.write(f"{h['epoch']},{h['d_loss']:.4f},{h['g_loss']:.4f},{h['colors']:.2f},{h['diversity']:.4f},{h['gate_score']:.4f}\\n")

generator.save_weights(os.path.join(ASSETS_DIR, "generator_final.weights.h5"))
ema_generator.save_weights(os.path.join(ASSETS_DIR, "generator_ema_final.weights.h5"))
print(f"Training complete in {(time.time() - t_start)/60:.1f} minutes! Best Gate Checkpoint and history saved.")
'''

nb["cells"][7]["source"] = [line + "\n" for line in cell7_code.split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated Cell 7 in 03_Discrete_SAGAN_Production.ipynb!")

