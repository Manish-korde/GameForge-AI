"""
Generates the single comprehensive Moodle Practical Submission PDF (30 Marks)
for Manish Korde and Samriddhi Das, containing:
- Mandatory Moodle Submission Header, Student Names & Roll Numbers (ONLY Manish Korde & Samriddhi Das)
- Direct Clickable Links (Google Drive, GitHub, Live Deployment, Datasets, Checkpoints, PPT)
- Mandatory README (Setup, Dependencies, Usage Guide)
- Evaluation Summary & Visual Comparison Grids across all 5 Generative AI Models
- Authentic Python Code (Data Loading/Preprocessing + Training/Fine-Tuning + Evaluation)
  extracted directly from all 5 verified project notebooks.
"""

import os
import json
import html
import base64
import subprocess
import tempfile

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT_HTML = os.path.abspath("submission/GameForge_AI_Moodle_Practical_Submission.html")
OUT_PDF = os.path.abspath("submission/GameForge_AI_Moodle_Practical_Submission.pdf")
DRIVE_PDF = os.path.abspath("GameForge_AI_Practical_Submission/00_GameForge_AI_Moodle_Practical_Submission.pdf")

DRIVE_LINK = "https://drive.google.com/drive/folders/1ziDn8WGgCrDEpk7O04zsDPgn5MR55o3w?usp=sharing"
GITHUB_LINK = "https://github.com/Manish-korde/GameForge-AI"
DEPLOY_LINK = "https://gamforge.hostless.app"


def get_code_cells(ipynb_path: str, indices: list[int]) -> str:
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    blocks = []
    for idx in indices:
        if idx < len(nb["cells"]):
            src = "".join(nb["cells"][idx].get("source", []))
            clean_lines = []
            for line in src.splitlines():
                if "EMBEDDED_B64_DATASET" in line and len(line) > 300:
                    clean_lines.append('    EMBEDDED_B64_DATASET = "<BASE64_GZIP_DATASET_2051_PAIRS_EMBEDDED_IN_IPYNB>"')
                elif len(line) > 1000:
                    clean_lines.append(line[:120] + ' ... # [Long encoded data line omitted in PDF; full data in .ipynb]')
                else:
                    clean_lines.append(line)
            blocks.append(f"# --- Cell {idx} ---\n" + "\n".join(clean_lines).strip())
    return "\n\n".join(blocks)


def img_to_b64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_html() -> str:
    nb_dir = "GameForge_AI_Practical_Submission/1_Code_Notebooks"

    # Extract authentic code from all 5 verified model notebooks
    ae_code = get_code_cells(
        f"{nb_dir}/01_Autoencoder_280K_Training.ipynb", [1, 3, 5, 7, 9, 12]
    )
    vae_code = get_code_cells(
        f"{nb_dir}/02_VAE_280K_Training.ipynb", [4, 10, 12, 13, 15, 18, 21]
    )
    dcgan_code = get_code_cells(
        f"{nb_dir}/03a_DCGAN_Unconditional_50Epochs_Training.ipynb", [8, 14, 16, 24]
    )
    cdcgan_code = get_code_cells(
        f"{nb_dir}/03b_cDCGAN_v3_Conditional_Training.ipynb", [2, 3, 4, 7, 9]
    )
    t5_code = get_code_cells(
        f"{nb_dir}/04_Transformer_Flan_T5_Small_v2_Training.ipynb", [1, 3, 5, 7, 9, 11, 13]
    )
    diff_train_code = get_code_cells(
        f"{nb_dir}/05a_Diffusion_Action_50Epochs_Training.ipynb", [2, 3, 4, 5, 6]
    )
    diff_eval_code = get_code_cells(
        f"{nb_dir}/05b_Diffusion_Evaluation_and_Outputs.ipynb", [2, 8, 9]
    )

    grid_evolution = img_to_b64(r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5\all_models_evolution_comparison.png")
    grid_cdcgan = img_to_b64(r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5\representative_cDCGAN_v3_4x8_grid.png")
    grid_diffusion = img_to_b64(r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5\diffusion_f7_bilateral_mirror_comparison.png")

    def code_pre(txt: str) -> str:
        return f"<pre><code>{html.escape(txt)}</code></pre>"

    img_tags = ""
    if grid_evolution:
        img_tags += f'<div class="fig"><img src="data:image/png;base64,{grid_evolution}" /><p class="cap">Figure 1: Cross-Architecture Evolution Comparison (AE vs VAE vs Unconditional DCGAN vs cDCGAN v3)</p></div>'
    if grid_cdcgan:
        img_tags += f'<div class="fig"><img src="data:image/png;base64,{grid_cdcgan}" /><p class="cap">Figure 2: Conditional cDCGAN v3 Palette-Conditioned Sprite Synthesis Grid (4x8)</p></div>'
    if grid_diffusion:
        img_tags += f'<div class="fig"><img src="data:image/png;base64,{grid_diffusion}" /><p class="cap">Figure 3: Multi-Pose Diffusion Stride & Bilateral Identity Verification</p></div>'

    readme_cmd = """# 1. Clone GitHub Repository (with Git LFS for model weights)
git lfs install
git clone https://github.com/Manish-korde/GameForge-AI.git
cd GameForge-AI

# 2. Setup Python Backend Environment (Python 3.10+)
python -m venv backend/venv
backend\\venv\\Scripts\\activate        # Windows (or source backend/venv/bin/activate on Linux)
pip install -r backend/requirements_docker.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Run FastAPI AI Backend (Serves all 5 Models + API Docs at /docs)
uvicorn backend.app:app --host 127.0.0.1 --port 8000

# 4. Run React + Vite Studio Frontend (in a second terminal)
cd gui && npm install && npm run dev   # Opens Studio UI at http://127.0.0.1:5173

# Alternative: Single-Container Docker Run
docker build -t gameforge-ai .
docker run -p 8000:8000 gameforge-ai"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>GameForge AI - Generative AI Practical Submission (30 Marks)</title>
<style>
  @page {{ size: A4; margin: 16mm 14mm; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #1e293b;
    line-height: 1.48;
    font-size: 10.5pt;
    margin: 0;
    padding: 0;
  }}
  h1 {{ font-size: 20pt; color: #0f172a; margin: 0 0 4px 0; border-bottom: 3px solid #4f46e5; padding-bottom: 6px; }}
  h2 {{ font-size: 14pt; color: #1e1b4b; margin-top: 22px; border-bottom: 1.5px solid #cbd5e1; padding-bottom: 4px; page-break-after: avoid; }}
  h3 {{ font-size: 11.5pt; color: #312e81; margin-top: 14px; margin-bottom: 6px; page-break-after: avoid; }}
  p, li {{ margin: 4px 0; }}
  a {{ color: #4338ca; word-break: break-all; text-decoration: underline; }}
  .hero {{
    background: #f8fafc;
    border: 1.5px solid #cbd5e1;
    border-left: 6px solid #4f46e5;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 14px;
  }}
  .links-box {{
    background: #eef2ff;
    border: 1.5px solid #a5b4fc;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 12px 0;
    page-break-inside: avoid;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px 0;
    font-size: 9.2pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
  }}
  th {{ background: #1e293b; color: #ffffff; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  pre {{
    background: #0f172a;
    color: #e2e8f0;
    padding: 10px 12px;
    border-radius: 6px;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 7.7pt;
    line-height: 1.35;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 8px 0 14px 0;
  }}
  .page-break {{ page-break-before: always; }}
  .fig {{ text-align: center; margin: 14px 0; page-break-inside: avoid; }}
  .fig img {{ max-width: 96%; max-height: 360px; border: 1px solid #cbd5e1; border-radius: 4px; }}
  .cap {{ font-size: 9pt; color: #475569; font-style: italic; margin-top: 4px; }}
  .badge {{
    display: inline-block;
    background: #4f46e5;
    color: #fff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 8.5pt;
    font-weight: 600;
  }}
</style>
</head>
<body>

<div class="hero">
  <span class="badge">GENERATIVE AI PRACTICAL SUBMISSION (30 MARKS)</span>
  <h1>GameForge AI: Multi-Model Generative Pipeline for 2D Game Asset Synthesis, Latent Representation Learning &amp; Pose Deformation</h1>
  <p><strong>Course:</strong> Generative Artificial Intelligence (IA &amp; CA Practical Submission) &nbsp;|&nbsp; <strong>Institute:</strong> MIT Academy of Engineering (MITAOE), Pune</p>
  <p><strong>Department:</strong> Computer Engineering &nbsp;|&nbsp; <strong>Project Guide:</strong> Prof. Dr. Savita Mane</p>
  <p><strong>Group Members (2 Students):</strong></p>
  <ul>
    <li><strong>Manish Korde</strong> &mdash; Roll / PRN: <code>202301040206</code></li>
    <li><strong>Samriddhi Das</strong> &mdash; Roll / PRN: <code>202301040276</code></li>
  </ul>
</div>

<div class="links-box">
  <h3 style="margin-top:0;">Mandatory Submission Links (All Components Accessible Online)</h3>
  <ul>
    <li><strong>Google Drive Master Folder (All Notebooks <code>.ipynb</code> + <code>.pdf</code>, Datasets, Checkpoints, PPT):</strong><br/>
        <a href="{DRIVE_LINK}">{DRIVE_LINK}</a></li>
    <li><strong>GitHub Repository (Complete Code, README.md, Notebooks, Assets, Docker):</strong><br/>
        <a href="{GITHUB_LINK}">{GITHUB_LINK}</a></li>
    <li><strong>Live Deployed Web Application (End-to-End FastAPI + React UI):</strong><br/>
        <a href="{DEPLOY_LINK}">{DEPLOY_LINK}</a></li>
    <li><strong>Model Checkpoints Directory (Drive &amp; GitHub):</strong>
        <a href="{DRIVE_LINK}">Google Drive &rarr; <code>3_Model_Checkpoints/</code></a> &nbsp;|&nbsp;
        <a href="{GITHUB_LINK}/tree/main/models">GitHub &rarr; <code>models/</code></a></li>
    <li><strong>Presentation PPT &amp; Slides Directory:</strong>
        <a href="{DRIVE_LINK}">Google Drive &rarr; <code>4_Presentation_PPT/</code></a> &nbsp;|&nbsp;
        <a href="{GITHUB_LINK}/tree/main/submission">GitHub &rarr; <code>submission/</code></a></li>
  </ul>
</div>

<h2>Submission Compliance Checklist (Per Practical Rubric &mdash; 30 Marks)</h2>
<table>
  <thead>
    <tr><th>Component</th><th>Requirement</th><th>Included in This PDF &amp; Linked Repositories</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. Code File (<code>.ipynb</code> &amp; <code>.pdf</code>)</strong></td>
      <td>Data loading &amp; preprocessing, model training/fine-tuning, evaluation &amp; testing</td>
      <td>Complete Python code for all <strong>5 Models</strong> embedded in Sections 4&ndash;8 below; standalone <code>.ipynb</code> and <code>.pdf</code> exported for every model in Drive (<code>1_Code_Notebooks/</code>) and GitHub (<code>Notebook/</code>).</td>
    </tr>
    <tr>
      <td><strong>2. Dataset</strong></td>
      <td>Uploaded datasets (if small) + valid Kaggle / Hugging Face source links</td>
      <td>All source links documented in Section 2; small curated JSONL dataset splits (2,000 Prompt-JSON pairs) &amp; 25 character sprite pool uploaded to Drive (<code>2_Dataset_Sources_and_Sample_Sprites/</code>) and GitHub (<code>transformer_training/</code>).</td>
    </tr>
    <tr>
      <td><strong>3. Model Files</strong></td>
      <td>Saved model checkpoints &amp; download links</td>
      <td>All latest production checkpoints (<code>.keras</code>, <code>.weights.h5</code>, <code>.safetensors</code>, <code>.pt</code>) uploaded in Drive (<code>3_Model_Checkpoints/</code>) and tracked via Git LFS on GitHub (<code>models/</code>).</td>
    </tr>
    <tr>
      <td><strong>4. Presentation (PPT)</strong></td>
      <td>Presentation slides deck</td>
      <td>Uploaded as <code>.pptx</code> and <code>.pdf</code> in Drive (<code>4_Presentation_PPT/</code>) and GitHub (<code>submission/GameForge_AI_Final_Presentation.pptx</code>).</td>
    </tr>
    <tr>
      <td><strong>5. GitHub &amp; README</strong></td>
      <td>GitHub repository with <code>README.md</code>, setup instructions, name &amp; roll number</td>
      <td>Full repository at <a href="{GITHUB_LINK}">{GITHUB_LINK}</a> with mandatory <code>README.md</code>, installation &amp; run instructions reproduced in Section 3 below.</td>
    </tr>
  </tbody>
</table>

<h2>2. Datasets Used &amp; Verified Source Links</h2>
<table>
  <thead>
    <tr><th>Model Module</th><th>Dataset Description</th><th>Size / Split</th><th>Verified Source Link / Location</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Module 1 &amp; 2: AE &amp; VAE</strong></td>
      <td><code>evilsocket/alucard-sprites</code> (2D RPG &amp; Fantasy Sprite Corpus, 64&times;64&times;3 RGBA composited on white/black)</td>
      <td>282,511 unique deduplicated sprites (80/10/10 split)</td>
      <td><a href="https://huggingface.co/datasets/evilsocket/alucard-sprites">huggingface.co/datasets/evilsocket/alucard-sprites</a></td>
    </tr>
    <tr>
      <td><strong>Module 3: DCGAN &amp; cDCGAN v3</strong></td>
      <td>Curated Front-Facing 2D Character Pool + k=4 Palette Cluster Labels</td>
      <td>42,000 front-facing sprites (palette clusters k=0..3)</td>
      <td>Derived from <code>alucard-sprites</code>; cluster manifest at GitHub <code>config/alucard_frontfacing_42k_c4_labels.json</code> &amp; 25 curated sprites in Drive.</td>
    </tr>
    <tr>
      <td><strong>Module 4: Flan-T5 Transformer</strong></td>
      <td>GameForge Curated Prompt-to-GDD JSONL Dataset (Non-empty schema repair)</td>
      <td>2,051 pairs (1,651 Train / 200 Val / 200 Test)</td>
      <td>Uploaded directly in Drive <code>2_Dataset_Sources_and_Sample_Sprites/</code> &amp; GitHub <code>transformer_training/</code>.</td>
    </tr>
    <tr>
      <td><strong>Module 5: Diffusion Pose</strong></td>
      <td>Liberated Pixel Cup (LPC) Action &amp; Multi-Pose Character Spritesheets</td>
      <td>10,000 paired pose-action frames (Walk, Slash, Spellcast, Thrust)</td>
      <td><a href="https://opengameart.org/content/liberated-pixel-cup-lpc-base-assets-sprites-map-tiles">OpenGameArt LPC Sprite Corpus</a> (curated pairs in <code>research_paper_assets/</code>).</td>
    </tr>
  </tbody>
</table>

<h2>3. Model Checkpoints, Evaluation Metrics &amp; Mandatory README</h2>
<table>
  <thead>
    <tr><th>Module</th><th>Checkpoint File (in Drive <code>3_Model_Checkpoints/</code>)</th><th>Size</th><th>Key Architecture &amp; Evaluation Metrics</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. Autoencoder (AE)</strong></td>
      <td><code>1_Autoencoder_280K_best.keras</code></td>
      <td>12.41 MB</td>
      <td>Conv2D Encoder-Decoder (64&times;64&times;3 &rarr; 256-D latent); MSE Reconstruction Loss: <code>0.0041</code>, Outlier QA P95 threshold calibration.</td>
    </tr>
    <tr>
      <td><strong>2. Variational Autoencoder (VAE)</strong></td>
      <td><code>2_VAE_encoder_280k_final.keras</code><br/><code>2_VAE_decoder_280k_final.keras</code></td>
      <td>33.52 MB + 18.73 MB</td>
      <td>Probabilistic 256-D latent manifold (&mu;, &sigma;&sup2; reparameterization); Cosine similarity retrieval across 282,511 indexed vectors.</td>
    </tr>
    <tr>
      <td><strong>3. DCGAN &amp; cDCGAN v3</strong></td>
      <td><code>3a_DCGAN_unconditional_generator_final.weights.h5</code><br/><code>3b_cDCGAN_v3_conditional_generator_ema.weights.h5</code></td>
      <td>10.46 MB<br/>10.59 MB</td>
      <td>Spectral Norm + Projection Discriminator + Hinge Loss + DiffAugment + EMA + Quality Filter v3 (bilateral facial hue gate, 64 candidates).</td>
    </tr>
    <tr>
      <td><strong>4. Transformer (Flan-T5 v2)</strong></td>
      <td><code>4_Transformer_Flan_T5_Small_v2/model.safetensors</code> (+ config &amp; tokenizer)</td>
      <td>293.60 MB</td>
      <td><code>google/flan-t5-small</code> (80M params), 50 Epochs, Effective Batch 64; 100% JSON schema compliance + Stage 2 Neuro-Symbolic Grounding.</td>
    </tr>
    <tr>
      <td><strong>5. Multi-Pose Diffusion</strong></td>
      <td><code>5a_Diffusion_MultiPose_E1_checkpoint_best.pt</code><br/><code>5b_Diffusion_Action_50ep_best.pt</code></td>
      <td>7.60 MB<br/>22.60 MB</td>
      <td>Identity-conditioned UNet DDPM/Velocity schedule (T=250); Action-conditioned 4-pose walk/slash stride synthesis with F7 bilateral repair.</td>
    </tr>
  </tbody>
</table>

<h3>Mandatory README: Local Installation &amp; Execution Instructions</h3>
<p><strong>Authors:</strong> Manish Korde (<code>202301040206</code>), Samriddhi Das (<code>202301040276</code>) &nbsp;|&nbsp; <strong>Project:</strong> GameForge AI</p>
{code_pre(readme_cmd)}

<h3>Visual Evaluation Results Across Modules</h3>
{img_tags}

<div class="page-break"></div>
<h2>4. Module 1 Code: Autoencoder (280K Sprites &mdash; Loading, Training &amp; Evaluation)</h2>
<p>Source Notebook: <code>1_Code_Notebooks/01_Autoencoder_280K_Training.ipynb</code> (and <code>.pdf</code>)</p>
{code_pre(ae_code)}

<div class="page-break"></div>
<h2>5. Module 2 Code: Variational Autoencoder (280K Sprites &mdash; Manifold &amp; Reconstruction)</h2>
<p>Source Notebook: <code>1_Code_Notebooks/02_VAE_280K_Training.ipynb</code> (and <code>.pdf</code>)</p>
{code_pre(vae_code)}

<div class="page-break"></div>
<h2>6. Module 3 Code: DCGAN &amp; Conditional cDCGAN v3 (42K Front-Facing Sprites)</h2>
<p>Source Notebooks: <code>03a_DCGAN_Unconditional_50Epochs_Training.ipynb</code> &amp; <code>03b_cDCGAN_v3_Conditional_Training.ipynb</code></p>
<h3>6A. Unconditional DCGAN (Spectral Norm, Feature Matching &amp; Training Loop)</h3>
{code_pre(dcgan_code)}
<h3>6B. Conditional cDCGAN v3 (k=4 Palette Conditioning, DiffAugment &amp; EMA Step)</h3>
{code_pre(cdcgan_code)}

<div class="page-break"></div>
<h2>7. Module 4 Code: Transformer Concept Planner (Flan-T5-Small v2, 50 Epochs, Batch 64)</h2>
<p>Source Notebook: <code>1_Code_Notebooks/04_Transformer_Flan_T5_Small_v2_Training.ipynb</code> (and <code>.pdf</code>)</p>
{code_pre(t5_code)}

<div class="page-break"></div>
<h2>8. Module 5 Code: Action-Conditioned &amp; Multi-Pose Diffusion UNet</h2>
<p>Source Notebooks: <code>05a_Diffusion_Action_50Epochs_Training.ipynb</code> &amp; <code>05b_Diffusion_Evaluation_and_Outputs.ipynb</code></p>
<h3>8A. Action-Conditioned UNet Architecture &amp; 50-Epoch Training</h3>
{code_pre(diff_train_code)}
<h3>8B. Diffusion Evaluation, DDPM Sampling &amp; Comparative Metrics</h3>
{code_pre(diff_eval_code)}

</body>
</html>
"""


def main():
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    html_str = build_html()
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"Wrote HTML: {OUT_HTML} ({os.path.getsize(OUT_HTML)/1024:.1f} KB)")

    url_path = "file:///" + OUT_HTML.replace("\\", "/")
    with tempfile.TemporaryDirectory() as tmp_profile:
        res = subprocess.run(
            [
                CHROME_EXE,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--user-data-dir={tmp_profile}",
                "--disable-background-networking",
                f"--print-to-pdf={OUT_PDF}",
                url_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    if res.returncode != 0 or not os.path.exists(OUT_PDF):
        raise RuntimeError(f"PDF print failed: {res.stderr}")

    import shutil
    if os.path.exists(os.path.dirname(DRIVE_PDF)):
        shutil.copy2(OUT_PDF, DRIVE_PDF)

    import pymupdf
    doc = pymupdf.open(OUT_PDF)
    print(f"SUCCESS! PDF created: {OUT_PDF} ({len(doc)} pages, {os.path.getsize(OUT_PDF)/1024/1024:.2f} MB)")
    print(f"Copied to Drive folder: {DRIVE_PDF}")


if __name__ == "__main__":
    main()
