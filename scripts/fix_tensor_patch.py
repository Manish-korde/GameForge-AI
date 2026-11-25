path = 'backend/multi_pose_diffusion_service.py'
with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

old_fn = """    def apply_feathered_bilateral_patch_tensor(self, f7_tensor: torch.Tensor, f3_tensor: torch.Tensor) -> torch.Tensor:
        \"\"\"Experimental Option: Applies bilateral mirror correction to F7 right-leg apex using flipped F3.\"\"\"
        f3_flipped = torch.flip(f3_tensor, dims=[-1])
        corrected = f7_tensor.clone()
        corrected[:, 41:64, :] = f3_flipped[:, 41:64, :]
        for i, r in enumerate(range(37, 41)):
            weight = (i + 1) / 5.0
            corrected[:, r, :] = (1.0 - weight) * f7_tensor[:, r, :] + weight * f3_flipped[:, r, :]
        return corrected"""

new_fn = """    def apply_feathered_bilateral_patch_tensor(self, f7_tensor: torch.Tensor, f3_tensor: torch.Tensor) -> torch.Tensor:
        \"\"\"Experimental Option: Applies bilateral mirror correction to F7 right-leg apex using flipped F3.\"\"\"
        f3_flipped = torch.flip(f3_tensor, dims=[-1])
        corrected = f7_tensor.clone()
        if corrected.ndim == 4:
            corrected[:, :, 41:64, :] = f3_flipped[:, :, 41:64, :]
            for i, r in enumerate(range(37, 41)):
                weight = (i + 1) / 5.0
                corrected[:, :, r, :] = (1.0 - weight) * f7_tensor[:, :, r, :] + weight * f3_flipped[:, :, r, :]
        else:
            corrected[:, 41:64, :] = f3_flipped[:, 41:64, :]
            for i, r in enumerate(range(37, 41)):
                weight = (i + 1) / 5.0
                corrected[:, r, :] = (1.0 - weight) * f7_tensor[:, r, :] + weight * f3_flipped[:, r, :]
        return corrected"""

if old_fn not in code:
    old_fn = old_fn.replace('\n', '\r\n')
    new_fn = new_fn.replace('\n', '\r\n')

assert old_fn in code, "old_fn not found"
code = code.replace(old_fn, new_fn, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(code)

print("Fixed apply_feathered_bilateral_patch_tensor successfully!")
