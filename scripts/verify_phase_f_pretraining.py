"""
Phase F: Pre-training Validation & Dataset Construction
Verifies:
1. 200 character IDs in candidate_crops
2. Exactly 9 front-facing frames per character (r02_c00 to r02_c08)
3. Split: 160 train characters, 40 val characters (no leakage)
4. Exact sample counts: 1,440 train, 360 val, 1,800 total
5. Checkpoint SHA-256 hash of Refined C2
6. Saves phase_F_experiment_E1/dataset_manifest.csv
"""

import os
import sys
import hashlib
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_DIR = os.path.join(ROOT_DIR, 'GameForge_AI_Phase2_audit', 'gameforge_phase2_1_audit', 'data', 'candidate_crops', 'action')
EXP_D_DIR = os.path.join(ROOT_DIR, 'corrective_experiment_D')
PHASE_F_DIR = os.path.join(ROOT_DIR, 'phase_F_experiment_E1')

def main():
    print("=" * 70)
    print("PHASE F: PRE-TRAINING VALIDATION & DATASET MANIFEST CONSTRUCTION")
    print("=" * 70)
    
    # 1. Verify Refined C2 Checkpoint
    ref_ckpt_path = os.path.join(EXP_D_DIR, 'refined_C2', 'checkpoint_best.pt')
    assert os.path.exists(ref_ckpt_path), f"Checkpoint not found: {ref_ckpt_path}"
    with open(ref_ckpt_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"Refined C2 Checkpoint Path: {ref_ckpt_path}")
    print(f"File Size: {os.path.getsize(ref_ckpt_path):,} bytes")
    print(f"SHA-256: {sha256}")
    expected_sha256 = "64f1a4cf1f859d66978a22119c23eac677da1f8990f480568d1b5345808d0f9e"
    assert sha256 == expected_sha256, f"Hash mismatch! Expected {expected_sha256}, got {sha256}"
    print(">> Refined C2 Checkpoint Hash Verified MATCH!\n")
    
    # 2. Check all 200 characters
    all_files = set(os.listdir(CANDIDATE_DIR))
    manifest_records = []
    
    print("Verifying 200 character IDs and 9 South frames per character...")
    for c_idx in range(200):
        cid = f"C_{c_idx:03d}"
        split = "train" if c_idx < 160 else "val"
        
        # Neutral reference: r02_c00
        ref_filename = f"action_row{c_idx:06d}_r02_c00.png"
        ref_path = os.path.join(CANDIDATE_DIR, ref_filename)
        assert ref_filename in all_files, f"Missing reference file: {ref_filename}"
        
        # Verify RGBA & shape
        im_ref = Image.open(ref_path)
        assert im_ref.size == (64, 64), f"Invalid size {im_ref.size} in {ref_path}"
        assert im_ref.mode == 'RGBA', f"Invalid mode {im_ref.mode} in {ref_path}"
        
        # 1. Idle sample: action=0, pose=0, target=r02_c00
        manifest_records.append({
            'sample_id': f"{cid}_idle_p0",
            'character_id': cid,
            'split': split,
            'action_id': 0,
            'action_name': 'idle',
            'pose_id': 0,
            'pose_name': 'idle_stance',
            'ref_file': ref_filename,
            'target_file': ref_filename,
            'ref_path': ref_path,
            'target_path': ref_path,
        })
        
        # 2-9. Walk samples: action=1, pose=0..7, target=r02_c01..c08
        for p_idx in range(8):
            target_filename = f"action_row{c_idx:06d}_r02_c{p_idx+1:02d}.png"
            target_path = os.path.join(CANDIDATE_DIR, target_filename)
            assert target_filename in all_files, f"Missing walk target: {target_filename}"
            
            im_tgt = Image.open(target_path)
            assert im_tgt.size == (64, 64) and im_tgt.mode == 'RGBA'
            
            manifest_records.append({
                'sample_id': f"{cid}_walk_f{p_idx+1}_p{p_idx}",
                'character_id': cid,
                'split': split,
                'action_id': 1,
                'action_name': 'walk',
                'pose_id': p_idx,
                'pose_name': f"walk_frame_{p_idx+1}",
                'ref_file': ref_filename,
                'target_file': target_filename,
                'ref_path': ref_path,
                'target_path': target_path,
            })
            
    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv = os.path.join(PHASE_F_DIR, 'dataset_manifest.csv')
    df_manifest.to_csv(manifest_csv, index=False)
    print(f">> Dataset Manifest Created: {manifest_csv}")
    
    # Verification checks
    train_count = len(df_manifest[df_manifest['split'] == 'train'])
    val_count = len(df_manifest[df_manifest['split'] == 'val'])
    total_count = len(df_manifest)
    
    train_chars = df_manifest[df_manifest['split'] == 'train']['character_id'].nunique()
    val_chars = df_manifest[df_manifest['split'] == 'val']['character_id'].nunique()
    
    print("\nDataset Verification Summary:")
    print(f"  Total Characters: {train_chars + val_chars} (160 train, 40 val)")
    print(f"  Training Samples: {train_count} (Expected: 1,440)")
    print(f"  Validation Samples: {val_count} (Expected: 360)")
    print(f"  Total Samples: {total_count} (Expected: 1,800)")
    
    assert train_chars == 160 and val_chars == 40
    assert train_count == 1440 and val_count == 360 and total_count == 1800
    
    # Check split hygiene (zero character overlap)
    train_char_set = set(df_manifest[df_manifest['split'] == 'train']['character_id'])
    val_char_set = set(df_manifest[df_manifest['split'] == 'val']['character_id'])
    overlap = train_char_set.intersection(val_char_set)
    assert len(overlap) == 0, f"Identity leakage detected: {overlap}"
    print(">> Split Hygiene Confirmed: ZERO identity overlap between train and val cohorts!")
    
    print("\nPhase 0 Pre-Training Validation PASSED COMPLETELY!\n")

if __name__ == '__main__':
    main()

