import asyncio
import os
import sys
import json
import base64
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app, generate_diffusion_endpoint, generate_walk_cycle_endpoint, get_diffusion_status, MultiPoseDiffusionRequest, WalkCycleRequest
from fastapi import HTTPException

# Test with a real character crop
ref_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..',
    'GameForge_AI_Phase2_audit', 'gameforge_phase2_1_audit', 'data', 'candidate_crops', 'action',
    'action_row000160_r02_c00.png'
))
with open(ref_path, 'rb') as f:
    ref_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

async def main():
    print("--- 1. Testing GET /generate/diffusion/status ---")
    status = get_diffusion_status()
    print("Status:", status['status'], "| Checkpoint:", status['checkpoint_sha256'][:16])
    assert status['status'] == 'ready'
    assert status['checkpoint_sha256'] == '10d33c5b2e7ad4d6f173f92fee2ee47101597bfadb1a677f6ee59d8bb7a1bc25'

    print("\n--- 2. Testing POST /generate/diffusion (Idle) ---")
    req_idle = MultiPoseDiffusionRequest(action="idle", pose_id=0, seed=42, steps=25, reference_b64=ref_b64)
    res_idle = await generate_diffusion_endpoint(req_idle)
    print("Idle Success:", res_idle['action'], res_idle['pose_name'], f"{res_idle['latency_ms']}ms")
    assert res_idle['status'] == 'success'
    assert res_idle['action'] == 'idle'
    assert res_idle['pose_id'] == 0

    print("\n--- 3. Testing POST /generate/diffusion (Walk F3) ---")
    req_f3 = MultiPoseDiffusionRequest(action="walk", pose_id=2, seed=42, steps=25, reference_b64=ref_b64)
    res_f3 = await generate_diffusion_endpoint(req_f3)
    print("Walk F3 Success:", res_f3['action'], res_f3['pose_name'], f"{res_f3['latency_ms']}ms")
    assert res_f3['status'] == 'success'
    assert res_f3['action'] == 'walk'
    assert res_f3['pose_id'] == 2

    print("\n--- 4. Testing Input Validation (Rejection of invalid actions/poses) ---")
    # Unsupported action: attack
    try:
        req_bad = MultiPoseDiffusionRequest(action="attack", pose_id=0, reference_b64=ref_b64)
        await generate_diffusion_endpoint(req_bad)
        print("FAIL: Accepted 'attack'")
    except HTTPException as e:
        print(f"PASS: Rejected 'attack' with HTTP {e.status_code}: {e.detail}")
        assert e.status_code == 400

    # Invalid pose for idle
    try:
        req_bad_idle = MultiPoseDiffusionRequest(action="idle", pose_id=1, reference_b64=ref_b64)
        await generate_diffusion_endpoint(req_bad_idle)
        print("FAIL: Accepted idle pose 1")
    except HTTPException as e:
        print(f"PASS: Rejected idle pose 1 with HTTP {e.status_code}: {e.detail}")
        assert e.status_code == 400

    # Invalid pose for walk
    try:
        req_bad_walk = MultiPoseDiffusionRequest(action="walk", pose_id=8, reference_b64=ref_b64)
        await generate_diffusion_endpoint(req_bad_walk)
        print("FAIL: Accepted walk pose 8")
    except HTTPException as e:
        print(f"PASS: Rejected walk pose 8 with HTTP {e.status_code}: {e.detail}")
        assert e.status_code == 400

    # Missing reference
    try:
        req_missing = MultiPoseDiffusionRequest(action="walk", pose_id=0, reference_b64=None)
        await generate_diffusion_endpoint(req_missing)
        print("FAIL: Accepted missing reference")
    except HTTPException as e:
        print(f"PASS: Rejected missing reference with HTTP {e.status_code}: {e.detail}")
        assert e.status_code == 400

    print("\nALL BACKEND API TESTS PASSED!")

if __name__ == '__main__':
    asyncio.run(main())

