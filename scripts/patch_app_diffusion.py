path = 'backend/app.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update MultiPoseDiffusionRequest
t1 = "class MultiPoseDiffusionRequest(BaseModel):\n    action: str = \"walk\"\n    pose_id: int = 0\n    seed: Optional[int] = None\n    steps: Optional[int] = 50\n    reference_b64: Optional[str] = None"
r1 = "class MultiPoseDiffusionRequest(BaseModel):\n    action: str = \"walk\"\n    pose_id: int = 0\n    seed: Optional[int] = None\n    steps: Optional[int] = 50\n    reference_b64: Optional[str] = None\n    enable_f7_patch: Optional[bool] = False"

if t1 not in content:
    t1 = t1.replace('\n', '\r\n')
    r1 = r1.replace('\n', '\r\n')

assert t1 in content, "t1 not found"
content = content.replace(t1, r1, 1)

# 2. Update generate_diffusion_endpoint
t2 = "            seed=req.seed,\n            steps=req.steps or 50\n        )"
r2 = "            seed=req.seed,\n            steps=req.steps or 50,\n            enable_f7_patch=req.enable_f7_patch or False\n        )"

if t2 not in content:
    t2 = t2.replace('\n', '\r\n')
    r2 = r2.replace('\n', '\r\n')

assert t2 in content, "t2 not found"
content = content.replace(t2, r2, 1)

# 3. Update WalkCycleRequest
t3 = "class WalkCycleRequest(BaseModel):\n    reference_b64: Optional[str] = None\n    seed: Optional[int] = 42\n    steps: Optional[int] = 50"
r3 = "class WalkCycleRequest(BaseModel):\n    reference_b64: Optional[str] = None\n    seed: Optional[int] = 42\n    steps: Optional[int] = 50\n    enable_f7_patch: Optional[bool] = False"

if t3 not in content:
    t3 = t3.replace('\n', '\r\n')
    r3 = r3.replace('\n', '\r\n')

assert t3 in content, "t3 not found"
content = content.replace(t3, r3, 1)

# 4. Update generate_walk_cycle_endpoint
t4 = "            seed=req.seed if req.seed is not None else 42,\n            steps=req.steps or 50\n        )"
r4 = "            seed=req.seed if req.seed is not None else 42,\n            steps=req.steps or 50,\n            enable_f7_patch=req.enable_f7_patch or False\n        )"

if t4 not in content:
    t4 = t4.replace('\n', '\r\n')
    r4 = r4.replace('\n', '\r\n')

assert t4 in content, "t4 not found"
content = content.replace(t4, r4, 1)

# 5. Update ActionDiffusionRequest
t5 = "class ActionDiffusionRequest(BaseModel):\n    action: Optional[str] = \"walk\"\n    pose_id: Optional[int] = 0\n    reference_b64: Optional[str] = None\n    steps: Optional[int] = 50\n    seed: Optional[int] = None"
r5 = "class ActionDiffusionRequest(BaseModel):\n    action: Optional[str] = \"walk\"\n    pose_id: Optional[int] = 0\n    reference_b64: Optional[str] = None\n    steps: Optional[int] = 50\n    seed: Optional[int] = None\n    enable_f7_patch: Optional[bool] = False"

if t5 not in content:
    t5 = t5.replace('\n', '\r\n')
    r5 = r5.replace('\n', '\r\n')

assert t5 in content, "t5 not found"
content = content.replace(t5, r5, 1)

# 6. Update action_diffusion_generate_endpoint
t6 = "            seed=req.seed,\n            steps=req.steps or 50\n        )"
r6 = "            seed=req.seed,\n            steps=req.steps or 50,\n            enable_f7_patch=req.enable_f7_patch or False\n        )"

if t6 not in content:
    t6 = t6.replace('\n', '\r\n')
    r6 = r6.replace('\n', '\r\n')

assert t6 in content, "t6 not found"
content = content.replace(t6, r6, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully updated backend/app.py with enable_f7_patch!")
