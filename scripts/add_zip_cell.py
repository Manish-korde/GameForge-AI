import json

nb_path = "Notebook/training/03_cDCGAN_k4_WarmStart_18Epochs.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_10_source = [
    "# 10. Package & Download Artifacts (Checkpoints, Final Weights, Samples, Logs)\n",
    "import os, sys, shutil\n",
    "from IPython.display import FileLink, display\n",
    "\n",
    "OUTPUT_ZIP = \"/kaggle/working/4_cDCGAN_k4_v1.zip\" if os.path.exists(\"/kaggle\") else \"./4_cDCGAN_k4_v1.zip\"\n",
    "TARGET_DIR = BASE_DIR\n",
    "\n",
    "print(\"=\" * 80)\n",
    "print(f\"PACKAGING ARTIFACTS FROM: {TARGET_DIR}\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "# 1. Create Zip Archive\n",
    "zip_base = OUTPUT_ZIP.replace(\".zip\", \"\")\n",
    "archive_path = shutil.make_archive(zip_base, \"zip\", root_dir=TARGET_DIR)\n",
    "zip_size_mb = os.path.getsize(archive_path) / (1024 * 1024)\n",
    "\n",
    "print(f\"Zip created successfully: {archive_path}\")\n",
    "print(f\"Total Archive Size: {zip_size_mb:.2f} MB\\n\")\n",
    "\n",
    "# 2. List Archived Contents Summary\n",
    "print(\"Archive Summary:\")\n",
    "for root, dirs, files in os.walk(TARGET_DIR):\n",
    "    rel_root = os.path.relpath(root, TARGET_DIR)\n",
    "    indent = \"  \" * (0 if rel_root == \".\" else rel_root.count(os.sep) + 1)\n",
    "    folder_name = \"models/4_cDCGAN_k4_v1\" if rel_root == \".\" else os.path.basename(root)\n",
    "    print(f\"{indent}📂 {folder_name}/ ({len(files)} files)\")\n",
    "    for f in sorted(files):\n",
    "        f_size_kb = os.path.getsize(os.path.join(root, f)) / 1024\n",
    "        print(f\"{indent}   📄 {f} ({f_size_kb:.1f} KB)\")\n",
    "\n",
    "# 3. Environment-Aware Download Trigger\n",
    "print(\"\\n\" + \"=\" * 80)\n",
    "print(\"DOWNLOAD OPTIONS:\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "if \"google.colab\" in sys.modules:\n",
    "    print(\"Detected Google Colab environment. Triggering browser download...\")\n",
    "    from google.colab import files\n",
    "    files.download(archive_path)\n",
    "else:\n",
    "    print(\"Detected Kaggle / Jupyter environment.\")\n",
    "    print(f\"1. Available in Kaggle Output pane as: {os.path.basename(archive_path)}\")\n",
    "    print(\"2. Click the link below to download directly:\")\n",
    "    zip_rel = os.path.basename(archive_path) if os.path.exists(\"/kaggle\") else archive_path\n",
    "    display(FileLink(zip_rel))\n"
]

if len(nb["cells"]) > 10:
    nb["cells"][10]["source"] = cell_10_source
else:
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_10_source
    })

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Cell 10 appended successfully. Total cells:", len(nb["cells"]))
