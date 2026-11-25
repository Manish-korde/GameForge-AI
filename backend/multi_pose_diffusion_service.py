"""
GameForge AI — Multi-Pose Action & Pose Diffusion Service
Production Integration of Phase F2 Validated Velocity Diffusion UNet.

Capabilities:
- Pure Gaussian initialization (x_T ~ N(0, I))
- Velocity-aware 50-step reverse DDIM sampling
- Action conditioning: Idle (0), Walk (1)
- Pose conditioning: Idle (0), Walk F1-F8 (0..7)
- Exact SHA-256 verification against Phase F2 validated checkpoint
- Thread-safe singleton execution with CUDA and CPU fallback
"""

import os
import io
import time
import math
import base64
import hashlib
import threading
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-threading for CPU inference
torch.set_num_threads(8)

# ----------------------------------------------------------------------
# 1. DIFFUSION SCHEDULE (1000 Timesteps Cosine)
# ----------------------------------------------------------------------
TIMESTEPS = 1000

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# ----------------------------------------------------------------------
# 2. MODEL ARCHITECTURE (ActionAndPoseConditionedUNet)
# ----------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.silu = nn.SiLU()
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.silu(self.norm1(self.conv1(x)))
        time_emb = self.silu(self.time_mlp(t))
        h = h + time_emb[(..., ) + (None, ) * 2]
        h = self.silu(self.norm2(self.conv2(h)))
        return h

class ActionAndPoseConditionedUNet(nn.Module):
    def __init__(self, time_dim: int = 128, num_actions: int = 2, num_poses: int = 8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        self.action_embed = nn.Embedding(num_actions, time_dim)
        self.pose_embed = nn.Embedding(num_poses, time_dim)
        
        self.in_conv = nn.Conv2d(8, 32, 3, padding=1)
        self.b1 = Block(32, 64, time_dim)
        self.down1 = nn.MaxPool2d(2)
        self.b2 = Block(64, 128, time_dim)
        self.down2 = nn.MaxPool2d(2)
        self.mid = Block(128, 256, time_dim)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.b_up2 = Block(256, 128, time_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.b_up1 = Block(128, 64, time_dim)
        self.out_conv = nn.Conv2d(64, 4, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_ref: torch.Tensor, action_id: torch.Tensor, pose_id: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_t, x_ref], dim=1)
        time_emb = self.time_mlp(t)
        act_emb = self.action_embed(action_id)
        pose_emb = self.pose_embed(pose_id)
        combined_emb = time_emb + act_emb + pose_emb
        
        h0 = self.in_conv(x)
        h1 = self.b1(h0, combined_emb)
        d1 = self.down1(h1)
        h2 = self.b2(d1, combined_emb)
        d2 = self.down2(h2)
        m = self.mid(d2, combined_emb)
        u2 = self.up2(m)
        cat2 = torch.cat([u2, h2], dim=1)
        dec2 = self.b_up2(cat2, combined_emb)
        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, h1], dim=1)
        dec1 = self.b_up1(cat1, combined_emb)
        return self.out_conv(dec1)

# ----------------------------------------------------------------------
# 3. MODULAR MULTI-POSE DIFFUSION SERVICE (SINGLETON)
# ----------------------------------------------------------------------
class MultiPoseDiffusionService:
    EXPECTED_SHA256 = "10d33c5b2e7ad4d6f173f92fee2ee47101597bfadb1a677f6ee59d8bb7a1bc25"
    EXPECTED_PARAMS = 1987044
    
    ACTION_MAP = {
        "idle": 0,
        "0": 0,
        0: 0,
        "walk": 1,
        "1": 1,
        1: 1
    }
    
    POSE_NAMES = {
        (0, 0): "Idle (Neutral Stance)",
        (1, 0): "Walk F1 (Neutral Passing)",
        (1, 1): "Walk F2 (Left Forward Plant)",
        (1, 2): "Walk F3 (Left Stride Apex)",
        (1, 3): "Walk F4 (Left Return Phase)",
        (1, 4): "Walk F5 (Right Lift Passing)",
        (1, 5): "Walk F6 (Right Forward Plant)",
        (1, 6): "Walk F7 (Right Stride Apex)",
        (1, 7): "Walk F8 (Neutral Return)"
    }

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.lock = threading.Lock()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if checkpoint_path is None:
            candidates = [
                os.path.join(root_dir, 'models', '5_Diffusion_LPC_v1', 'checkpoint_best.pt'),
                os.path.join(root_dir, 'research_paper_assets', '06_sprite_deformation_and_pose_animation', '11_phase_F_experiment_E1_multi_pose', 'checkpoint_best.pt'),
                os.path.join(root_dir, 'legacy_root_experiment_runs', 'phase_F_experiment_E1', 'checkpoint_best.pt'),
                os.path.join(root_dir, 'phase_F_experiment_E1', 'checkpoint_best.pt'),
            ]
            self.checkpoint_path = candidates[0]
            for c in candidates:
                if os.path.exists(c):
                    self.checkpoint_path = c
                    break
        else:
            self.checkpoint_path = checkpoint_path
            
        self.betas = cosine_beta_schedule(TIMESTEPS).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(self.device)
        
        self.model: Optional[ActionAndPoseConditionedUNet] = None
        self.checkpoint_sha256: str = ""
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None
        
        self._initialize_model()

    def _initialize_model(self):
        """Loads and verifies the model checkpoint with strict SHA-256 and param checks."""
        try:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")
            
            with open(self.checkpoint_path, 'rb') as f:
                self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest().lower()
                
            if self.checkpoint_sha256 != self.EXPECTED_SHA256:
                raise ValueError(
                    f"Checkpoint SHA-256 mismatch! Expected {self.EXPECTED_SHA256}, got {self.checkpoint_sha256}."
                )
                
            model = ActionAndPoseConditionedUNet().to(self.device)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            param_count = sum(p.numel() for p in model.parameters())
            if param_count != self.EXPECTED_PARAMS:
                raise ValueError(f"Parameter count mismatch! Expected {self.EXPECTED_PARAMS}, got {param_count}.")
                
            self.model = model
            self.is_loaded = True
            self.load_error = None
            print(f"MultiPoseDiffusionService initialized successfully on {self.device}.")
            print(f"Loaded: {self.checkpoint_path} (SHA-256: {self.checkpoint_sha256})")
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            print(f"ERROR: MultiPoseDiffusionService failed to initialize: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.is_loaded else "error",
            "model_name": "ActionAndPoseConditionedUNet",
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parameter_count": self.EXPECTED_PARAMS,
            "supported_actions": {
                "idle": {"action_id": 0, "valid_poses": [0]},
                "walk": {"action_id": 1, "valid_poses": list(range(8))}
            },
            "diffusion_steps_default": 50,
            "initialization": "pure_gaussian (x_T ~ N(0, I))",
            "sampler": "reverse_ddim_velocity",
            "error": self.load_error
        }

    def validate_action_and_pose(self, action: Any, pose_id: Any) -> Tuple[int, int, str]:
        """Validates action and pose strictly according to Phase F2 specifications."""
        action_clean = str(action).strip().lower()
        if action_clean not in self.ACTION_MAP:
            raise ValueError(
                f"Unsupported action '{action}'. Supported actions are: 'idle' (0) and 'walk' (1). "
                "Actions such as 'attack', 'run', or 'jump' are not supported."
            )
            
        action_id = self.ACTION_MAP[action_clean]
        
        try:
            pose_id = int(pose_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid pose_id '{pose_id}'. Must be an integer.")
            
        if action_id == 0: # Idle
            if pose_id != 0:
                raise ValueError(
                    f"Invalid pose_id '{pose_id}' for Idle stance. Idle only supports pose_id=0."
                )
        elif action_id == 1: # Walk
            if pose_id < 0 or pose_id > 7:
                raise ValueError(
                    f"Invalid pose_id '{pose_id}' for Walk. Walk supports pose_id 0 through 7 (F1 through F8)."
                )
                
        pose_name = self.POSE_NAMES.get((action_id, pose_id), f"Action_{action_id}_Pose_{pose_id}")
        return action_id, pose_id, pose_name

    def preprocess_reference(self, reference_input: Any) -> Tuple[torch.Tensor, Image.Image]:
        """
        Converts input (base64 data URL, raw base64, bytes, or PIL Image)
        into exact Phase F2 normalized tensor: (1, 4, 64, 64) in [-1.0, 1.0].
        """
        if reference_input is None:
            raise ValueError("Reference sprite image is required.")
            
        pil_img: Optional[Image.Image] = None
        
        if isinstance(reference_input, Image.Image):
            pil_img = reference_input.convert("RGBA")
        elif isinstance(reference_input, str):
            data_str = reference_input.strip()
            if data_str.startswith("http://") or data_str.startswith("https://"):
                # If localhost / 127.0.0.1, read file directly to prevent single-worker deadlock
                if ":8000/" in data_str:
                    path_part = data_str.split(":8000/", 1)[1].split("?")[0]
                    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    # Check direct path, or inside root
                    cand1 = os.path.join(root_dir, path_part.replace("/", os.sep))
                    cand2 = os.path.join(root_dir, "backend", path_part.replace("/", os.sep))
                    for c in [cand1, cand2]:
                        if os.path.exists(c):
                            try:
                                pil_img = Image.open(c).convert("RGBA")
                                break
                            except Exception:
                                pass
                if pil_img is None:
                    try:
                        import urllib.request
                        req = urllib.request.Request(data_str, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=4) as resp:
                            pil_img = Image.open(io.BytesIO(resp.read())).convert("RGBA")
                    except Exception as e:
                        raise ValueError(f"Failed to load reference image from URL: {e}")
            elif os.path.exists(data_str):
                try:
                    pil_img = Image.open(data_str).convert("RGBA")
                except Exception as e:
                    raise ValueError(f"Failed to open image file '{data_str}': {e}")
            else:
                if data_str.startswith("data:image"):
                    data_str = data_str.split(",", 1)[1]
                # Auto-pad missing base64 padding characters ('=')
                pad_len = len(data_str) % 4
                if pad_len != 0:
                    data_str += "=" * (4 - pad_len)
                try:
                    img_bytes = base64.b64decode(data_str)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 reference image: {e}")
        elif isinstance(reference_input, (bytes, bytearray)):
            try:
                pil_img = Image.open(io.BytesIO(reference_input)).convert("RGBA")
            except Exception as e:
                raise ValueError(f"Failed to parse reference image bytes: {e}")
        else:
            raise ValueError(f"Unsupported reference image format: {type(reference_input)}")
            
        if pil_img.size != (64, 64):
            pil_img = pil_img.resize((64, 64), Image.Resampling.BILINEAR)
            
        arr = np.array(pil_img, dtype=np.float32) / 127.5 - 1.0
        tensor_chw = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device) # (1, 4, 64, 64)
        return tensor_chw, pil_img

    def tensor_to_pil(self, tensor_1chw: torch.Tensor) -> Image.Image:
        """Converts generated (1, 4, 64, 64) clamped tensor to PIL RGBA Image."""
        t_np = ((tensor_1chw[0].cpu().clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(t_np, mode='RGBA')

    def pil_to_base64(self, pil_img: Image.Image, format: str = "PNG") -> str:
        buffered = io.BytesIO()
        pil_img.save(buffered, format=format)
        return f"data:image/{format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    def sample_ddim(
        self,
        ref_tensor: torch.Tensor,
        action_id: int,
        pose_id: int,
        steps: int = 50,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Executes reverse DDIM sampling from pure Gaussian noise (x_T ~ N(0, I)).
        Uses validated velocity prediction parameterization.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Multi-pose diffusion model is not loaded: {self.load_error}")
            
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
                
        # Pure Gaussian initialization
        x_cur = torch.randn(1, 4, 64, 64, device=self.device)
        
        step_ratio = TIMESTEPS // steps
        timesteps_seq = np.asarray(list(range(0, TIMESTEPS, step_ratio)))
        
        act_t = torch.tensor([action_id], dtype=torch.long, device=self.device)
        pose_t = torch.tensor([pose_id], dtype=torch.long, device=self.device)
        
        with torch.inference_mode():
            for step_idx in reversed(range(len(timesteps_seq))):
                t_idx = timesteps_seq[step_idx]
                t = torch.tensor([t_idx], device=self.device, dtype=torch.long)
                
                a_bar = self.alphas_bar[t_idx]
                a_bar_prev = self.alphas_bar[timesteps_seq[step_idx - 1]] if step_idx > 0 else torch.tensor(1.0, device=self.device)
                
                pred_v = self.model(x_cur, t, ref_tensor, act_t, pose_t)
                
                pred_x0 = torch.sqrt(a_bar) * x_cur - torch.sqrt(1.0 - a_bar) * pred_v
                pred_eps = torch.sqrt(1.0 - a_bar) * x_cur + torch.sqrt(a_bar) * pred_v
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                
                dir_xt = torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=0.0)) * pred_eps
                x_cur = torch.sqrt(a_bar_prev) * pred_x0 + dir_xt
                
        return torch.clamp(x_cur, -1.0, 1.0)


    def apply_feathered_bilateral_patch_tensor(self, f7_tensor: torch.Tensor, f3_tensor: torch.Tensor) -> torch.Tensor:
        """Experimental Option: Applies bilateral mirror correction to F7 right-leg apex using flipped F3."""
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

    def generate_single_pose(
        self,
        reference_input: Any,
        action: Any,
        pose_id: Any,
        seed: Optional[int] = None,
        steps: int = 50,
        enable_f7_patch: bool = False
    ) -> Dict[str, Any]:
        """Public API method for generating a single action/pose sprite."""
        t0 = time.time()
        
        action_id, pose_id, pose_name = self.validate_action_and_pose(action, pose_id)
        ref_tensor, ref_pil = self.preprocess_reference(reference_input)
        
        if steps not in (25, 50, 100):
            steps = 50 # Fallback to validated default
            
        with self.lock:
            gen_tensor = self.sample_ddim(ref_tensor, action_id, pose_id, steps=steps, seed=seed)
            if enable_f7_patch and action_id == 1 and pose_id == 6:
                f3_tensor = self.sample_ddim(ref_tensor, action_id=1, pose_id=2, steps=steps, seed=seed)
                gen_tensor = self.apply_feathered_bilateral_patch_tensor(gen_tensor, f3_tensor)
            
        gen_pil = self.tensor_to_pil(gen_tensor)
        elapsed_sec = time.time() - t0
        
        return {
            "status": "success",
            "action": "walk" if action_id == 1 else "idle",
            "action_id": action_id,
            "pose_id": pose_id,
            "pose_name": pose_name,
            "seed": seed,
            "steps": steps,
            "enable_f7_patch": enable_f7_patch,
            "dimensions": [64, 64],
            "image_b64": self.pil_to_base64(gen_pil),
            "reference_b64": self.pil_to_base64(ref_pil),
            "latency_ms": round(elapsed_sec * 1000.0, 2),
            "latency_sec": round(elapsed_sec, 3),
            "device": str(self.device),
            "checkpoint_sha256": self.checkpoint_sha256
        }

    def generate_walk_cycle(
        self,
        reference_input: Any,
        seed: Optional[int] = 42,
        steps: int = 50,
        enable_f7_patch: bool = False
    ) -> Dict[str, Any]:
        """
        Generates the full 8-frame walk cycle (F1 through F8) from one reference sprite.
        Returns all 8 individual frames, a horizontal sprite sheet (512x64), and an animated GIF.
        """
        t0 = time.time()
        ref_tensor, ref_pil = self.preprocess_reference(reference_input)
        
        if steps not in (25, 50, 100):
            steps = 50
            
        frames: List[Dict[str, Any]] = []
        pil_frames: List[Image.Image] = []
        
        with self.lock:
            for p_idx in range(8):
                gen_t = self.sample_ddim(ref_tensor, action_id=1, pose_id=p_idx, steps=steps, seed=seed)
                p_img = self.tensor_to_pil(gen_t)
                if enable_f7_patch and p_idx == 6 and len(pil_frames) > 2:
                    p_img = self.apply_feathered_bilateral_patch_pil(p_img, pil_frames[2])
                pil_frames.append(p_img)
                pose_name = self.POSE_NAMES[(1, p_idx)]
                frames.append({
                    "pose_id": p_idx,
                    "pose_name": pose_name,
                    "image_b64": self.pil_to_base64(p_img)
                })
                
        # Composite horizontal sprite sheet (512 x 64)
        sheet_img = Image.new('RGBA', (8 * 64, 64), (0, 0, 0, 0))
        for idx, img in enumerate(pil_frames):
            sheet_img.paste(img, (idx * 64, 0))
        sheet_b64 = self.pil_to_base64(sheet_img)
        
        # Looping Animated GIF (125ms per frame = 8 fps)
        gif_io = io.BytesIO()
        pil_frames[0].save(
            gif_io,
            format='GIF',
            save_all=True,
            append_images=pil_frames[1:],
            duration=125,
            loop=0
        )
        gif_b64 = f"data:image/gif;base64,{base64.b64encode(gif_io.getvalue()).decode('utf-8')}"
        
        total_elapsed = time.time() - t0
        return {
            "status": "success",
            "action": "walk",
            "frame_count": 8,
            "seed": seed,
            "steps": steps,
            "frames": frames,
            "sprite_sheet_b64": sheet_b64,
            "gif_b64": gif_b64,
            "dimensions": [512, 64],
            "total_latency_ms": round(total_elapsed * 1000.0, 2),
            "total_latency_sec": round(total_elapsed, 3),
            "device": str(self.device),
            "checkpoint_sha256": self.checkpoint_sha256
        }

# Singleton instance
multi_pose_diffusion_service = MultiPoseDiffusionService()

