import os

path = 'backend/multi_pose_diffusion_service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

patch_code = '''
    def apply_feathered_bilateral_patch_tensor(self, f7_tensor: torch.Tensor, f3_tensor: torch.Tensor) -> torch.Tensor:
        """Experimental Option: Applies bilateral mirror correction to F7 right-leg apex using flipped F3."""
        f3_flipped = torch.flip(f3_tensor, dims=[-1])
        corrected = f7_tensor.clone()
        corrected[:, 41:64, :] = f3_flipped[:, 41:64, :]
        for i, r in enumerate(range(37, 41)):
            weight = (i + 1) / 5.0
            corrected[:, r, :] = (1.0 - weight) * f7_tensor[:, r, :] + weight * f3_flipped[:, r, :]
        return corrected

    def apply_feathered_bilateral_patch_pil(self, f7_img: Image.Image, f3_img: Image.Image) -> Image.Image:
        f7_arr = np.array(f7_img, dtype=np.float32)
        f3_arr = np.array(f3_img, dtype=np.float32)
        f3_flipped = np.fliplr(f3_arr)
        corrected = f7_arr.copy()
        corrected[41:64, :] = f3_flipped[41:64, :]
        for i, r in enumerate(range(37, 41)):
            weight = (i + 1) / 5.0
            corrected[r, :] = (1.0 - weight) * f7_arr[r, :] + weight * f3_flipped[r, :]
        return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))
'''

# 1. Insert helper methods and enable_f7_patch into generate_single_pose
target1 = "    def generate_single_pose(\n        self,\n        reference_input: Any,\n        action: Any,\n        pose_id: Any,\n        seed: Optional[int] = None,\n        steps: int = 50\n    ) -> Dict[str, Any]:"
replacement1 = patch_code + "\n    def generate_single_pose(\n        self,\n        reference_input: Any,\n        action: Any,\n        pose_id: Any,\n        seed: Optional[int] = None,\n        steps: int = 50,\n        enable_f7_patch: bool = False\n    ) -> Dict[str, Any]:"

if target1 not in content:
    # Try with CRLF
    target1 = target1.replace('\n', '\r\n')
    replacement1 = replacement1.replace('\n', '\r\n')

assert target1 in content, "target1 not found"
content = content.replace(target1, replacement1, 1)

# 2. Update sampling inside generate_single_pose
target2 = "        with self.lock:\n            gen_tensor = self.sample_ddim(ref_tensor, action_id, pose_id, steps=steps, seed=seed)"
replacement2 = "        with self.lock:\n            gen_tensor = self.sample_ddim(ref_tensor, action_id, pose_id, steps=steps, seed=seed)\n            if enable_f7_patch and action_id == 1 and pose_id == 6:\n                f3_tensor = self.sample_ddim(ref_tensor, action_id=1, pose_id=2, steps=steps, seed=seed)\n                gen_tensor = self.apply_feathered_bilateral_patch_tensor(gen_tensor, f3_tensor)"

if target2 not in content:
    target2 = target2.replace('\n', '\r\n')
    replacement2 = replacement2.replace('\n', '\r\n')

assert target2 in content, "target2 not found"
content = content.replace(target2, replacement2, 1)

# 3. Add enable_f7_patch to return dict
target3 = '            "steps": steps,\n            "dimensions": [64, 64],'
replacement3 = '            "steps": steps,\n            "enable_f7_patch": enable_f7_patch,\n            "dimensions": [64, 64],'

if target3 not in content:
    target3 = target3.replace('\n', '\r\n')
    replacement3 = replacement3.replace('\n', '\r\n')

assert target3 in content, "target3 not found"
content = content.replace(target3, replacement3, 1)

# 4. Update generate_walk_cycle signature
target4 = "    def generate_walk_cycle(\n        self,\n        reference_input: Any,\n        seed: Optional[int] = 42,\n        steps: int = 50\n    ) -> Dict[str, Any]:"
replacement4 = "    def generate_walk_cycle(\n        self,\n        reference_input: Any,\n        seed: Optional[int] = 42,\n        steps: int = 50,\n        enable_f7_patch: bool = False\n    ) -> Dict[str, Any]:"

if target4 not in content:
    target4 = target4.replace('\n', '\r\n')
    replacement4 = replacement4.replace('\n', '\r\n')

assert target4 in content, "target4 not found"
content = content.replace(target4, replacement4, 1)

# 5. Update loop in generate_walk_cycle
target5 = "                gen_t = self.sample_ddim(ref_tensor, action_id=1, pose_id=p_idx, steps=steps, seed=seed)\n                p_img = self.tensor_to_pil(gen_t)\n                pil_frames.append(p_img)"
replacement5 = "                gen_t = self.sample_ddim(ref_tensor, action_id=1, pose_id=p_idx, steps=steps, seed=seed)\n                p_img = self.tensor_to_pil(gen_t)\n                if enable_f7_patch and p_idx == 6 and len(pil_frames) > 2:\n                    p_img = self.apply_feathered_bilateral_patch_pil(p_img, pil_frames[2])\n                pil_frames.append(p_img)"

if target5 not in content:
    target5 = target5.replace('\n', '\r\n')
    replacement5 = replacement5.replace('\n', '\r\n')

assert target5 in content, "target5 not found"
content = content.replace(target5, replacement5, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully updated multi_pose_diffusion_service.py")
