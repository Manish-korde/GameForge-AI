"""
GameForge AI - Complete Practical Submission Package Builder (30 Marks)
Assembles all 5 models' training/evaluation notebooks (.ipynb + .pdf),
datasets & links, latest model checkpoints, presentation, and audit reports
into `GameForge_AI_Practical_Submission/` ready for Google Drive upload.
"""

import os
import sys
import shutil
import subprocess

TARGET_DIR = os.path.abspath("GameForge_AI_Practical_Submission")
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def create_structure():
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    subdirs = [
        "1_Code_Notebooks",
        "2_Dataset_Sources_and_Sample_Sprites",
        "2_Dataset_Sources_and_Sample_Sprites/frontfacing_25_character_pool",
        "3_Model_Checkpoints",
        "3_Model_Checkpoints/4_Transformer_Flan_T5_Small_v2",
        "4_Presentation_PPT",
        "5_Audit_Reports_and_Visual_Grids",
    ]
    for sd in subdirs:
        os.makedirs(os.path.join(TARGET_DIR, sd), exist_ok=True)
    print("Created clean directory structure for GameForge_AI_Practical_Submission.")


def convert_ipynb_to_pdf(ipynb_path: str, pdf_path: str) -> bool:
    """Converts a Jupyter notebook to HTML via nbconvert and then to PDF via headless Chrome."""
    import json
    import re
    import tempfile

    # Create a temporary copy for PDF rendering so giant embedded data cells (>12KB) don't stall layout
    tmp_ipynb = ipynb_path[:-6] + "_pdfrender.ipynb"
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if len(src) > 12000:
                lines = src.splitlines(keepends=True)
                truncated = "".join(lines[:35]) + "\n# ... [Truncated massive embedded dataset/data block for PDF readability; full data is in the .ipynb file] ...\n" + "".join(lines[-15:])
                cell["source"] = [truncated]
    with open(tmp_ipynb, "w", encoding="utf-8") as f:
        json.dump(nb, f)

    html_path = tmp_ipynb[:-6] + ".html"
    res_html = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", "--template", "classic", tmp_ipynb],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if os.path.exists(tmp_ipynb):
        os.remove(tmp_ipynb)

    if res_html.returncode != 0 or not os.path.exists(html_path):
        print(f"  [ERROR] nbconvert failed for {os.path.basename(ipynb_path)}: {res_html.stderr}")
        return False

    # Strip all <script> tags and external <link> tags so Chrome renders static HTML in <2s
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()
    html_content = re.sub(r'<script\b[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<link[^>]+href=["\']https?://[^"\']+["\'][^>]*>', '', html_content, flags=re.IGNORECASE)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    url_path = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    with tempfile.TemporaryDirectory() as tmp_profile:
        res_pdf = subprocess.run(
            [
                CHROME_EXE,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--user-data-dir={tmp_profile}",
                "--disable-background-networking",
                f"--print-to-pdf={os.path.abspath(pdf_path)}",
                url_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    # Remove intermediate HTML to keep folder clean
    if os.path.exists(html_path):
        os.remove(html_path)

    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10_000:
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"  [OK] Exported PDF: {os.path.basename(pdf_path)} ({size_kb:.1f} KB)")
        return True

    print(f"  [ERROR] Chrome PDF failed for {os.path.basename(ipynb_path)}: {res_pdf.stderr}")
    return False


def copy_and_export_notebooks():
    dest = os.path.join(TARGET_DIR, "1_Code_Notebooks")

    # Complete per-model notebooks: (source_path, clean_submission_name)
    notebooks = [
        ("Notebook/training/280k_dataset_autoencoder.ipynb", "01_Autoencoder_280K_Training.ipynb"),
        ("Notebook/training/280k_dataset_VAE.ipynb", "02_VAE_280K_Training.ipynb"),
        ("Notebook/training/02_DCGAN_280K_Run2_Perfected_50Epochs.ipynb", "03a_DCGAN_Unconditional_50Epochs_Training.ipynb"),
        ("Notebook/training/03_cDCGAN_k4_WarmStart_18Epochs.ipynb", "03b_cDCGAN_v3_Conditional_Training.ipynb"),
        ("Notebook/training/05_GameForge_Flan_T5_Small_Complete_Kaggle.ipynb", "04_Transformer_Flan_T5_Small_v2_Training.ipynb"),
        (
            "research_paper_assets/06_sprite_deformation_and_pose_animation/14_historical_50ep_and_150ep_baseline/kaggle_action_conditioned_diffusion_50ep.ipynb",
            "05a_Diffusion_Action_50Epochs_Training.ipynb",
        ),
        ("Notebook/evaluation/11_Diffusion_Action_Deformation_Evaluation_and_Outputs.ipynb", "05b_Diffusion_Evaluation_and_Outputs.ipynb"),
    ]

    for src, clean_name in notebooks:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing required notebook: {src}")
        dst_ipynb = os.path.join(dest, clean_name)
        dst_pdf = os.path.join(dest, clean_name[:-6] + ".pdf")
        shutil.copy2(src, dst_ipynb)
        print(f"Copied notebook: {clean_name}")
        ok = convert_ipynb_to_pdf(dst_ipynb, dst_pdf)
        if not ok:
            raise RuntimeError(f"Failed to export PDF for {clean_name}")

    # Include supporting documentation
    if os.path.exists("TRANSFORMER_QNA.md"):
        shutil.copy2("TRANSFORMER_QNA.md", dest)
        print("Copied supporting doc: TRANSFORMER_QNA.md")


def copy_datasets():
    dest_docs = os.path.join(TARGET_DIR, "2_Dataset_Sources_and_Sample_Sprites")
    shutil.copy2("submission/DATASET_SOURCES_AND_LINKS.md", dest_docs)

    for jf in [
        "submission/flan_t5_game_design_50_dataset.jsonl",
        "transformer_training/train_repaired.jsonl",
        "transformer_training/val_repaired.jsonl",
        "transformer_training/test_repaired.jsonl",
    ]:
        if os.path.exists(jf):
            shutil.copy2(jf, dest_docs)
            print(f"Copied dataset split: {os.path.basename(jf)}")

    src_sprites = "backend/frontfacing_25" if os.path.exists("backend/frontfacing_25") else "gui/public/frontfacing_25"
    dest_sprites = os.path.join(dest_docs, "frontfacing_25_character_pool")
    if os.path.exists(src_sprites):
        for item in os.listdir(src_sprites):
            sp = os.path.join(src_sprites, item)
            if os.path.isfile(sp):
                shutil.copy2(sp, dest_sprites)
        print(f"Copied {len(os.listdir(dest_sprites))} sample sprite files.")


def copy_models():
    dest = os.path.join(TARGET_DIR, "3_Model_Checkpoints")
    shutil.copy2("submission/MODEL_CHECKPOINTS_AND_MANIFEST.md", dest)

    flat_models = [
        ("models/1_AE_280k_v1/AE_280K_best.keras", "1_Autoencoder_280K_best.keras"),
        ("models/2_VAE_280k_v2/VAE_280K_Outputs/encoder_280k_final.keras", "2_VAE_encoder_280k_final.keras"),
        ("models/2_VAE_280k_v2/VAE_280K_Outputs/decoder_280k_final.keras", "2_VAE_decoder_280k_final.keras"),
        ("models/3_DCGAN_42k_v1/game_ready_assets/generator_weights_final.weights.h5", "3a_DCGAN_unconditional_generator_final.weights.h5"),
        ("models/5_final_cDCGAN_v3/cgenerator_ema_weights_final.weights.h5", "3b_cDCGAN_v3_conditional_generator_ema.weights.h5"),
        ("models/5_Diffusion_LPC_v1/checkpoint_best.pt", "5a_Diffusion_MultiPose_E1_checkpoint_best.pt"),
        ("models/5_Diffusion_LPC_v1/best_diffusion_50ep.pt", "5b_Diffusion_Action_50ep_best.pt"),
        ("models/5_Diffusion_LPC_v1/training_config.json", "5c_Diffusion_training_config.json"),
    ]
    for src, dst_name in flat_models:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing required checkpoint: {src}")
        shutil.copy2(src, os.path.join(dest, dst_name))
        size_mb = os.path.getsize(src) / (1024 * 1024)
        print(f"Copied checkpoint: {dst_name} ({size_mb:.2f} MB)")

    # Complete HuggingFace-compatible Transformer v2 directory
    t5_dest = os.path.join(dest, "4_Transformer_Flan_T5_Small_v2")
    t5_files = [
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_metrics.json",
        "evaluation_summary.json",
    ]
    for tf_name in t5_files:
        src = os.path.join("models/4_Transformer_2k_v1", tf_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing Transformer v2 file: {src}")
        shutil.copy2(src, os.path.join(t5_dest, tf_name))
        size_mb = os.path.getsize(src) / (1024 * 1024)
        print(f"Copied Transformer v2 file: 4_Transformer_Flan_T5_Small_v2/{tf_name} ({size_mb:.2f} MB)")


def copy_presentation():
    dest = os.path.join(TARGET_DIR, "4_Presentation_PPT")
    for pf in [
        "submission/GameForge_AI_Final_Presentation.pptx",
        "submission/GameForge_AI_Final_Presentation.pdf",
        "submission/PRESENTATION_SLIDES_AND_SPEAKER_NOTES.md",
    ]:
        if os.path.exists(pf):
            shutil.copy2(pf, dest)
    print("Copied presentation files to 4_Presentation_PPT.")


def copy_reports_and_grids():
    dest = os.path.join(TARGET_DIR, "5_Audit_Reports_and_Visual_Grids")
    shutil.copy2("submission/PRACTICAL_SUBMISSION_CHECKLIST.md", dest)
    if os.path.exists("DIFFUSION_RIGHT_LEG_STRIDE_SOLUTIONS.md"):
        shutil.copy2("DIFFUSION_RIGHT_LEG_STRIDE_SOLUTIONS.md", dest)

    artifacts = [
        r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5\diffusion_f7_bilateral_mirror_comparison.png",
        r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5\representative_cDCGAN_v3_4x8_grid.png",
        r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5\all_models_evolution_comparison.png",
    ]
    for art in artifacts:
        if os.path.exists(art):
            shutil.copy2(art, dest)
            print(f"Copied visual grid: {os.path.basename(art)}")


def write_root_submission_overview():
    content = """# GameForge AI: Practical Submission Package (30 Marks)
### Course: Generative AI Practical Evaluation | Academic Year 2026-2027
**Institute:** MIT Academy of Engineering (MITAOE), Pune  
**Department:** Department of Computer Engineering  
**Project Guide:** Prof. Dr. Savita Mane  

**Student Group Members:**
1. **Manish Korde** — PRN / Roll: `202301040206`
2. **Samriddhi Das** — PRN / Roll: `202301040276`
3. **Swayam Pagare** — PRN / Roll: `202301040248`

---

## Package Contents (All 5 Models Covered in Full)

### 1. `1_Code_Notebooks/` (Both `.ipynb` and `.pdf` for Every Model)
* `01_Autoencoder_280K_Training.ipynb` & `.pdf` — Autoencoder (280K dataset) data loading, training, reconstruction eval.
* `02_VAE_280K_Training.ipynb` & `.pdf` — Variational Autoencoder (280K dataset) KL-divergence + reconstruction training.
* `03a_DCGAN_Unconditional_50Epochs_Training.ipynb` & `.pdf` — Discrete Unconditional DCGAN (50 epochs).
* `03b_cDCGAN_v3_Conditional_Training.ipynb` & `.pdf` — Latest Conditional cDCGAN v3 (k=4 palette clusters, warm-start).
* `04_Transformer_Flan_T5_Small_v2_Training.ipynb` & `.pdf` — Flan-T5-Small v2 fine-tuning (50 epochs, batch 64, 2,000 JSONL pairs).
* `05a_Diffusion_Action_50Epochs_Training.ipynb` & `.pdf` — Action-Conditioned UNet Diffusion training (50 epochs).
* `05b_Diffusion_Evaluation_and_Outputs.ipynb` & `.pdf` — Diffusion evaluation, stride verification, and visual grids.
* `TRANSFORMER_QNA.md` — Technical defense & architecture documentation.

### 2. `2_Dataset_Sources_and_Sample_Sprites/`
* `DATASET_SOURCES_AND_LINKS.md` — Official Kaggle & Hugging Face dataset links.
* `train_repaired.jsonl`, `val_repaired.jsonl`, `test_repaired.jsonl` — Curated 2,000-pair Transformer dataset (80/10/10 split).
* `frontfacing_25_character_pool/` — 25 curated 64x64 character sprites used for pose deformation.

### 3. `3_Model_Checkpoints/` (All Latest Production Weights)
* `1_Autoencoder_280K_best.keras` (12.41 MB)
* `2_VAE_encoder_280k_final.keras` (33.52 MB) & `2_VAE_decoder_280k_final.keras` (18.73 MB)
* `3a_DCGAN_unconditional_generator_final.weights.h5` (10.46 MB)
* `3b_cDCGAN_v3_conditional_generator_ema.weights.h5` (10.59 MB)
* `4_Transformer_Flan_T5_Small_v2/` — Complete v2 directory (`model.safetensors` 293.6 MB, `config.json`, `tokenizer.json`, metrics).
* `5a_Diffusion_MultiPose_E1_checkpoint_best.pt` (7.60 MB) & `5b_Diffusion_Action_50ep_best.pt` (22.60 MB)
"""
    md_path = os.path.join(TARGET_DIR, "00_README_AND_SUBMISSION_GUIDE.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Created: 00_README_AND_SUBMISSION_GUIDE.md")


if __name__ == "__main__":
    print("Assembling complete Practical Submission folder...")
    create_structure()
    copy_and_export_notebooks()
    copy_datasets()
    copy_models()
    copy_presentation()
    copy_reports_and_grids()
    write_root_submission_overview()
    print(f"\nDONE! Clean package ready at:\n{TARGET_DIR}")
