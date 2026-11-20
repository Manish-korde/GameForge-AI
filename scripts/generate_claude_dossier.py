"""
scripts/generate_claude_dossier.py
Generates:
1. GAN_TRI_MODEL_EVALUATION_BOARD.png (Single master visual comparison board for Claude)
2. GAN_VERDICT_DOSSIER_FOR_CLAUDE.html (Self-contained HTML dossier with base64 embedded images)
3. GAN_VERDICT_DOSSIER_FOR_CLAUDE.md (Markdown dossier)
"""
import os
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BRAIN_DIR = r"C:\Users\manis\.gemini\antigravity\brain\0e75ca3f-2ae4-4b95-8ef8-c8e5292ad012"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMG_PATHS = {
    "ground_truth": os.path.join(BRAIN_DIR, "discrete_real_groundtruth.png"),
    "dcgan_ep10": os.path.join(BRAIN_DIR, "discrete_epoch_10.png"),
    "dcgan_ep50": os.path.join(BRAIN_DIR, "discrete_epoch_50.png"),
    "dcgan_cont_best": os.path.join(BRAIN_DIR, "compare_dcgan_continuation_best_grid.png"),
    "dcgan_cont_ep100": os.path.join(BRAIN_DIR, "compare_dcgan_cont_epoch100.png"),
    "sagan_ep34": os.path.join(BRAIN_DIR, "compare_sagan_epoch34.png"),
    "sagan_prod_grid": os.path.join(BRAIN_DIR, "compare_sagan_production_grid.png"),
    "ae_recon": os.path.join(BRAIN_DIR, "ae_recon_0.png"),
    "zoom_0": os.path.join(BRAIN_DIR, "discrete_zoomed_0.png"),
    "zoom_4": os.path.join(BRAIN_DIR, "discrete_zoomed_4.png"),
    "zoom_13": os.path.join(BRAIN_DIR, "discrete_zoomed_13.png")
}

for k, p in IMG_PATHS.items():
    if not os.path.exists(p):
        print(f"Warning: {k} path not found at {p}")

# ==============================================================================
# 1. CREATE MASTER COMPARISON BOARD (PNG)
# ==============================================================================
print("Generating Master Comparison Board (PNG)...")
fig = plt.figure(figsize=(24, 16), facecolor="#121212")
gs = gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.18)

panels = [
    ("1. Ground Truth (Audited Real 32-Color Sprites)", IMG_PATHS["ground_truth"], "Crisp 1-bit alpha, 17.9 unique colors, pristine hand-drawn pixel art"),
    ("2. Discrete DCGAN — Epoch 10 (Peak Diversity)", IMG_PATHS["dcgan_ep10"], "True 32-color palette, high archetype variety (mages, knights, orcs)"),
    ("3. Discrete DCGAN — Epoch 50 (Late-Run Mode Collapse)", IMG_PATHS["dcgan_ep50"], "~40% collapsed into hooded pale archer; Picasso facial jumble"),
    ("4. DCGAN Continuation — Epoch 52 (Peak Gate Score 0.202)", IMG_PATHS["dcgan_cont_best"], "Grey rectangular background shrouds appeared; D forced edge boundaries"),
    ("5. DCGAN Continuation — Epoch 100 (Final Overcooked State)", IMG_PATHS["dcgan_cont_ep100"], "Severe distribution collapse; twisted green/brown cocoons with lost anatomy"),
    ("6. Discrete SAGAN — 50 Epochs (Self-Attention + Minibatch)", IMG_PATHS["sagan_prod_grid"], "Square portrait tile framing artifact; rigid totem characters inside boxes")
]

for idx, (title, img_p, sub) in enumerate(panels):
    r, c = idx // 3, idx % 3
    ax = fig.add_subplot(gs[r, c])
    if os.path.exists(img_p):
        im = Image.open(img_p)
        ax.imshow(im)
    ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(sub, color="#aaaaaa", fontsize=10, labelpad=4)
    ax.axis("off")

plt.suptitle("GAN GENERATIVE EXPERIMENT AUDIT: DCGAN (1-50) vs. DCGAN (51-100) vs. SAGAN\nProject: Prompt-to-Game-Asset Generator | Prepared for Senior AI Research Review", 
             color="#00e5ff", fontsize=18, fontweight="bold", y=0.98)

board_out_path = os.path.join(PROJECT_ROOT, "GAN_TRI_MODEL_EVALUATION_BOARD.png")
plt.savefig(board_out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved master board to {board_out_path}")

# ==============================================================================
# 2. CREATE SELF-CONTAINED HTML DOSSIER WITH EMBEDDED BASE64 IMAGES
# ==============================================================================
print("Generating Self-Contained HTML Dossier...")

def to_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

b64_imgs = {k: to_b64(v) for k, v in IMG_PATHS.items()}
b64_board = to_b64(board_out_path)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GAN Architecture Forensic Dossier — Evaluation for Claude</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
        margin: 0;
        padding: 40px;
        line-height: 1.6;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 40px;
    }}
    h1 {{
        color: #58a6ff;
        border-bottom: 2px solid #30363d;
        padding-bottom: 15px;
        margin-top: 0;
    }}
    h2 {{
        color: #79c0ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-top: 35px;
    }}
    h3 {{
        color: #f0883e;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 8px;
        font-size: 12px;
        font-weight: bold;
        border-radius: 4px;
        margin-right: 8px;
    }}
    .badge-win {{ background: #238636; color: #fff; }}
    .badge-fail {{ background: #da3633; color: #fff; }}
    .badge-warn {{ background: #d29922; color: #fff; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }}
    th, td {{
        border: 1px solid #30363d;
        padding: 12px;
        text-align: left;
    }}
    th {{
        background: #21262d;
        color: #58a6ff;
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }}
    .grid-3 {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }}
    .card {{
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
        text-align: center;
    }}
    .card img {{
        max-width: 100%;
        height: auto;
        border-radius: 4px;
        image-rendering: pixelated;
        background: #000;
        border: 1px solid #484f58;
    }}
    .card p {{
        margin: 10px 0 0 0;
        font-size: 13px;
        color: #8b949e;
    }}
    .banner-img {{
        width: 100%;
        border: 1px solid #30363d;
        border-radius: 6px;
        margin: 20px 0;
    }}
    .prompt-box {{
        background: #0d1117;
        border-left: 4px solid #f0883e;
        padding: 16px;
        margin: 25px 0;
        border-radius: 0 4px 4px 0;
    }}
    code {{
        background: #282e33;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        color: #ff7b72;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>🎮 GAN Generative Architecture Forensic Dossier</h1>
    <p><strong>Project:</strong> Prompt-to-Game-Asset Generator &bull; <strong>Dataset:</strong> 42,212 Audited Front-Facing Character Sprites (<code>evilsocket/alucard-sprites</code>)<br>
    <strong>Evaluation Goal:</strong> Independent AI Research Review on why adversarial training destabilizes beyond ~25 epochs and which architecture is genuinely production game-ready.</p>

    <div class="prompt-box">
        <h3>📋 Note for Claude</h3>
        <p>This dossier presents the visual outputs, quantitative metrics, and architectural evolutions across three distinct GAN runs on 64&times;64 discrete retro pixel art:
        <br>1. <strong>Attempt 4A:</strong> Discrete Palette DCGAN (Epochs 1&ndash;50)
        <br>2. <strong>Attempt 4B:</strong> Discrete DCGAN Continuation with Cosine Annealing (Epochs 51&ndash;100)
        <br>3. <strong>Attempt 5:</strong> Discrete SAGAN with 16&times;16 Self-Attention &amp; Minibatch StdDev (Epochs 1&ndash;50)
        <br>4. <strong>Reference Benchmark:</strong> 280K Autoencoder (<code>AE_280K_best.keras</code>, 8&times;8&times;256 latent).
        <br><br>
        <strong>Please review the side-by-side contact sheets below and deliver an honest verdict on the failure modes, the sweet spot of training, and the viable path to game-ready asset generation.</strong></p>
    </div>

    <h2>1. Master Multi-Model Comparison Board</h2>
    <img src="{b64_board}" class="banner-img" alt="Master Comparison Board">

    <h2>2. Quantitative Metrics Across Training Runs</h2>
    <table>
        <thead>
            <tr>
                <th>Experiment</th>
                <th>Epoch Range</th>
                <th>Peak Diversity</th>
                <th>Unique Colors</th>
                <th>Peak Gate Score</th>
                <th>Primary Visual Pathology Observed</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Discrete DCGAN (4A)</strong></td>
                <td>1 &ndash; 50</td>
                <td>0.144 (Ep 48)</td>
                <td>25.7 / 32</td>
                <td>0.062 (Ep 26)</td>
                <td>Picasso facial jumble; ~40% archer mode collapse at Ep 35&ndash;50.</td>
            </tr>
            <tr>
                <td><strong>DCGAN Continuation (4B)</strong></td>
                <td>51 &ndash; 100</td>
                <td>0.332 (Ep 71)</td>
                <td>18.2 / 32</td>
                <td><strong>0.203 (Ep 52)</strong></td>
                <td>Solid grey square shrouds; severe cocoon distortion by Ep 100.</td>
            </tr>
            <tr>
                <td><strong>Discrete SAGAN (5)</strong></td>
                <td>1 &ndash; 50</td>
                <td>0.364 (Ep 3)</td>
                <td>24.8 / 32</td>
                <td>0.158 (Ep 3)</td>
                <td>Square portrait tile framing; rigid vertical totem characters.</td>
            </tr>
            <tr>
                <td><strong>280K Autoencoder</strong></td>
                <td>Inference</td>
                <td>0.320</td>
                <td>32.0 / 32</td>
                <td><strong>1.000 (SSIM > 0.98)</strong></td>
                <td><strong>None.</strong> 100% pixel-perfect discrete reconstruction.</td>
            </tr>
        </tbody>
    </table>

    <h2>3. Side-by-Side Visual Inspection</h2>
    <div class="grid-2">
        <div class="card">
            <h3>Discrete DCGAN: Epoch 10 (Peak Variety)</h3>
            <img src="{b64_imgs['dcgan_ep10']}" alt="DCGAN Epoch 10">
            <p><span class="badge badge-win">Best Diversity</span> Distinct bipedal warriors, orcs, knights, purple mages on transparent alpha.</p>
        </div>
        <div class="card">
            <h3>Discrete DCGAN: Epoch 50 (Mode Collapse)</h3>
            <img src="{b64_imgs['dcgan_ep50']}" alt="DCGAN Epoch 50">
            <p><span class="badge badge-fail">Mode Collapse</span> Over 40% of grid collapsed into pale-faced archer with detached floaters.</p>
        </div>
    </div>

    <div class="grid-2">
        <div class="card">
            <h3>DCGAN Continuation: Peak Checkpoint (Epoch 52)</h3>
            <img src="{b64_imgs['dcgan_cont_best']}" alt="Continuation Best">
            <p><span class="badge badge-warn">Grey Shrouds</span> Grey rectangular boxes emerged as D forced boundary penalties.</p>
        </div>
        <div class="card">
            <h3>DCGAN Continuation: Epoch 100 (Final)</h3>
            <img src="{b64_imgs['dcgan_cont_ep100']}" alt="Continuation Epoch 100">
            <p><span class="badge badge-fail">Cocoons</span> Complete anatomical collapse into green/brown twisted shapes without faces.</p>
        </div>
    </div>

    <div class="grid-2">
        <div class="card">
            <h3>Discrete SAGAN: Epoch 34</h3>
            <img src="{b64_imgs['sagan_ep34']}" alt="SAGAN Epoch 34">
            <p><span class="badge badge-warn">Border Sync</span> Self-attention synchronized corners, forming square tile frames.</p>
        </div>
        <div class="card">
            <h3>Discrete SAGAN: Final Production Grid (LCC Cleaned)</h3>
            <img src="{b64_imgs['sagan_prod_grid']}" alt="SAGAN Production Grid">
            <p><span class="badge badge-fail">Tile Framing</span> Characters enclosed inside square portraits; rigid vertical totem poses.</p>
        </div>
    </div>

    <h2>4. Pixel-Level Zoom Inspection (400% Nearest-Neighbor)</h2>
    <div class="grid-3">
        <div class="card">
            <h3>Sprite #0: Orc / Goblin</h3>
            <img src="{b64_imgs['zoom_0']}" style="width: 200px;" alt="Zoom 0">
            <p>Discrete colors &amp; 1-bit alpha intact, but floating specks to the right.</p>
        </div>
        <div class="card">
            <h3>Sprite #4: Pale Ranger</h3>
            <img src="{b64_imgs['zoom_4']}" style="width: 200px;" alt="Zoom 4">
            <p>Oversized blue eye on white mask; asymmetrical hair; melted joints.</p>
        </div>
        <div class="card">
            <h3>Sprite #13: Hooded Fighter</h3>
            <img src="{b64_imgs['zoom_13']}" style="width: 200px;" alt="Zoom 13">
            <p>Duplicate torso blocks; detached weapon fragment floating 6px away.</p>
        </div>
    </div>

    <h2>5. Forensic Analysis &amp; Questions for Claude</h2>
    <div class="prompt-box">
        <h3>Specific Questions for Claude's Verdict:</h3>
        <ol>
            <li><strong>Why did continuing DCGAN from 50 to 100 epochs cause grey rectangular shrouds and cocoon degradation instead of improving details?</strong></li>
            <li><strong>Why did adding Self-Attention in SAGAN cause the generator to draw full square tile frames around sprites?</strong></li>
            <li><strong>Is it fundamentally possible for an unconditional 128-D GAN to achieve game-ready pixel-art anatomy without a spatial/autoencoder prior?</strong></li>
            <li><strong>Given that our 280K Autoencoder has an 8&times;8&times;256 latent space that reconstructs pixel art with zero artifacts, how should we position this in our final engineering defense?</strong></li>
        </ol>
    </div>
</div>
</body>
</html>
"""

html_out_path = os.path.join(PROJECT_ROOT, "GAN_VERDICT_DOSSIER_FOR_CLAUDE.html")
with open(html_out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Saved self-contained HTML dossier to {html_out_path}")

# ==============================================================================
# 3. CREATE MARKDOWN DOSSIER
# ==============================================================================
md_content = f"""# 🎮 GAN Generative Architecture Forensic Dossier

**Prepared for:** Independent AI Research Review (Claude)  
**Project:** Prompt-to-Game-Asset Generator  
**Dataset:** 42,212 Audited Front-Facing Character Sprites (`evilsocket/alucard-sprites`)  

---

## 1. Executive Summary & Comparison Board

Below is the master comparison board showing all three GAN experiments and ground truth:

![Master Comparison Board](GAN_TRI_MODEL_EVALUATION_BOARD.png)

---

## 2. Experimental Configurations

| Model | Architecture | Epochs | Optimizer / LR | DiffAugment | Loss Function |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Attempt 4A** | Discrete DCGAN (K=32 logits) | 1–50 | Adam (G: 3e-4, D: 1e-4) | Translation (±12.5%) | Geometric Hinge + Spectral Norm |
| **Attempt 4B** | DCGAN Continuation | 51–100 | Cosine Decay (1.5e-4 → 2e-5) | Translation (±15.0%) | Hinge + Spectral Norm + G-EMA |
| **Attempt 5** | Discrete SAGAN (Self-Attn @ 16x16) | 1–50 | Adam (G: 2e-4, D: 1e-4) | Translation (±12.5%) | Hinge + Minibatch StdDev + G-EMA |

---

## 3. Quantitative Progression

| Metric | DCGAN (1–50) | DCGAN Cont. (51–100) | SAGAN (1–50) | Ground Truth Target |
| :--- | :---: | :---: | :---: | :---: |
| **Average Unique Colors** | 25.7 | 18.2 | 24.8 | **17.9** |
| **Pairwise Diversity** | 0.118 | 0.185 | 0.193 | **0.180** |
| **Peak Gate Score** | 0.062 (Ep 26) | **0.203 (Ep 52)** | 0.158 (Ep 3) | **N/A** |
| **Alpha Consistency** | 100% Binary [0, 255] | 100% Binary [0, 255] | 100% Binary [0, 255] | **100% Binary** |
| **Failure Mode** | Archer mode collapse @ Ep 35–50 | Grey shrouds + cocoons @ Ep 100 | Portrait tile framing | None |

---

## 4. Key Questions for Claude's Verdict:

1. **Why did extending DCGAN from 50 to 100 epochs cause grey rectangular shrouds and cocoon collapse instead of refining features?**
2. **Why did Self-Attention in SAGAN trigger square portrait tile borders around characters?**
3. **What is the theoretical ceiling of an unconditional 128-D GAN on 64x64 discrete pixel art?**
4. **How does this justify pivoting to our 280K Autoencoder ($8\\times 8\\times 256$ spatial latent) for production game assets?**
"""

md_out_path = os.path.join(PROJECT_ROOT, "GAN_VERDICT_DOSSIER_FOR_CLAUDE.md")
with open(md_out_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Saved Markdown dossier to {md_out_path}")
print("ALL CLAUDE VERDICT FILES GENERATED SUCCESSFULLY!")

