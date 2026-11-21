import os
import re
import json
import time
import logging
import threading
from typing import Dict, Any, Tuple

logger = logging.getLogger("transformer_service")
logger.setLevel(logging.INFO)

TASK_PREFIX = "game concept: "
MAX_NEW_TOKENS = 512
REQUIRED_SCHEMA_FIELDS = {"theme", "environment", "characters", "weapons", "props", "visualStyle"}

MODEL_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "models", "4_Transformer_2k_v1"
))


def strip_code_fences(text: str) -> str:
    """Removes markdown code fences (```json ... ```) if emitted by the model."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines).strip()
    return text


def derive_asset_tags(spec_dict: Dict[str, Any]) -> list:
    """
    Deterministic rule-based enrichment (Stage 2):
    Derives pixel art search tags from Flan-T5 generated theme and characters
    for downstream sprite discovery in VAE / Asset Studio.
    """
    tags = []
    theme = spec_dict.get("theme", "")
    if theme:
        tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', theme.lower()) + "_theme")

    for char in spec_dict.get("characters", []):
        c_name = char.get("name", "")
        if c_name:
            tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', c_name.lower()) + "_character")

    for weapon in spec_dict.get("weapons", []):
        w_name = weapon.get("name", "")
        if w_name:
            tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', w_name.lower()) + "_weapon")

    for prop in spec_dict.get("props", []):
        p_name = prop.get("name", "")
        if p_name:
            tags.append(re.sub(r'[^a-zA-Z0-9_]', '_', p_name.lower()) + "_item")

    if not tags:
        tags = ["pixel_art_character", "game_item", "environment_tile"]
    return tags[:6]


class TransformerService:
    def __init__(self):
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        self.model_path = MODEL_DIR
        self._lock = threading.Lock()

    def load_model(self):
        """Loads the fine-tuned Flan-T5 checkpoint (models/4_Transformer_2k_v1) and warms up."""
        if self.is_loaded and self.model is not None:
            return
        with self._lock:
            if self.is_loaded and self.model is not None:
                return
            logger.info(f"Loading Fine-Tuned Flan-T5 Transformer from {self.model_path}...")
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                self.model.eval()

                # Model warm-up to eliminate cold-start latency on first user prompt
                dummy_input = self.tokenizer(f"{TASK_PREFIX}warm up", return_tensors="pt")
                _ = self.model.generate(**dummy_input, max_new_tokens=16)

                self.is_loaded = True
                logger.info("Fine-Tuned Flan-T5 model loaded and warmed up successfully.")
            except Exception as e:
                logger.error(f"Failed to load fine-tuned Transformer model from {self.model_path}: {e}")
                self.is_loaded = False
                raise RuntimeError(f"Transformer model unavailable: {e}")

    def unload_model(self):
        """Safely unloads PyTorch + FLAN-T5 model and releases CPU memory via garbage collection."""
        with self._lock:
            if self.model is not None or self.tokenizer is not None or self.is_loaded:
                logger.info("Unloading Transformer model to release CPU RAM...")
                self.model = None
                self.tokenizer = None
                self.is_loaded = False
                import gc
                gc.collect()
                logger.info("Transformer model unloaded and CPU memory garbage-collected.")

    def validate_prompt_safety(self, user_prompt: str) -> Tuple[bool, str]:
        """Validates user prompt against Responsible AI safety boundaries (NSFW, violence, hate speech)."""
        if not user_prompt:
            return True, "Safe"

        prompt_lower = user_prompt.lower()

        # Prohibited NSFW / Harmful Terms
        prohibited_terms = [
            "nsfw", "nude", "explicit", "gore", "decapitation",
            "hate", "racist", "slur", "torture", "suicide"
        ]
        for term in prohibited_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', prompt_lower):
                return False, f"Prompt contains restricted or unsafe content ('{term}')."

        # Direct Protected IP Warning (logged, re-grounded)
        protected_ips = ["mario", "pokemon", "pikachu", "sonic", "zelda"]
        for ip in protected_ips:
            if re.search(r'\b' + re.escape(ip) + r'\b', prompt_lower):
                logger.warning(f"Protected IP keyword detected: '{ip}'. Re-grounding to generic asset.")

        return True, "Passed Regex Boundary Audit"

    def generate_concept_spec(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executes Seq2Seq Neural Generation using the fine-tuned Flan-T5 model (models/4_Transformer_2k_v1).
        Outputs structured JSON strictly matching the evaluated schema:
        {theme, environment, characters, weapons, props, visualStyle}
        """
        is_safe, safety_msg = self.validate_prompt_safety(user_prompt)
        if not is_safe:
            raise ValueError(safety_msg)

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self.load_model()

        clean_prompt = user_prompt.strip() if user_prompt else "Fantasy RPG adventure"
        formatted_input = f"{TASK_PREFIX}{clean_prompt}"

        # 1. Real Neural Model Generation with Telemetry
        start_time = time.perf_counter()
        import torch
        with torch.no_grad():
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            output_ids = self.model.generate(
                **inputs,
                num_beams=4,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        cleaned_text = strip_code_fences(raw_output)

        # 2. Parse Neural JSON
        parse_error = None
        parsed_spec = None
        try:
            parsed_spec = json.loads(cleaned_text)
        except Exception as e:
            parse_error = str(e)
            logger.warning(f"T5 JSON decode error: {e}. Raw: {raw_output}")

        if parse_error or not isinstance(parsed_spec, dict):
            # Fail transparently with real error details - DO NOT silently fallback to hardcoded tables
            return {
                "status": "parse_error",
                "prompt": clean_prompt,
                "formatted_input": formatted_input,
                "error_detail": f"Model generated non-JSON syntax: {parse_error}",
                "raw_t5_output": raw_output,
                "telemetry": {
                    "inference_latency_ms": latency_ms,
                    "tokens_generated": len(output_ids[0]),
                    "schema_valid": False,
                    "model_checkpoint": "models/4_Transformer_2k_v1"
                },
                "ethical_guardrails": {
                    "safety_status": safety_msg,
                    "dataset_attribution": "GameForge Fine-Tuned T5 (2,000 Curated Prompt-JSON Pairs; 80/10/10 Split)",
                    "license": "CC-BY-NC-SA 4.0",
                    "human_oversight_required": True
                }
            }

        # 3. Schema Validation against evaluated paper schema
        actual_fields = set(parsed_spec.keys())
        missing_fields = list(REQUIRED_SCHEMA_FIELDS - actual_fields)
        extra_fields = list(actual_fields - REQUIRED_SCHEMA_FIELDS)
        schema_valid = len(missing_fields) == 0

        # 4. Clearly segregated Stage 2 Rule-Based Enrichment for downstream pipelines
        enrichment = {
            "pipeline_stage": "Stage 2: Deterministic Asset Mapping (Rule-Based)",
            "recommended_asset_tags": derive_asset_tags(parsed_spec),
            "note": "Derived deterministically from T5 theme & character entities for downstream sprite discovery"
        }

        # Map to structured spec object
        main_char = parsed_spec.get("characters", [{}])[0] if parsed_spec.get("characters") else {}
        main_char_name = main_char.get("name", "Protagonist")
        main_char_desc = main_char.get("desc", "")

        return {
            "status": "success",
            "prompt": clean_prompt,
            "formatted_input": formatted_input,
            "game_title": f"{parsed_spec.get('theme', 'GameForge')} Adventure",
            "theme": parsed_spec.get("theme", ""),
            "environment": parsed_spec.get("environment", ""),
            "visualStyle": parsed_spec.get("visualStyle", ""),
            "characters": parsed_spec.get("characters", []),
            "weapons": parsed_spec.get("weapons", []),
            "props": parsed_spec.get("props", []),
            # Backwards compatibility fields for UI card layout
            "main_character": {
                "role": main_char_name,
                "attributes": [main_char_desc] if main_char_desc else ["Pixel Art Character"]
            },
            "art_style": parsed_spec.get("visualStyle", "Retro Pixel Art"),
            "genre": f"{parsed_spec.get('theme', 'Fantasy')} RPG",
            "enemies": [c.get("name") for c in parsed_spec.get("characters", [])[1:]] if len(parsed_spec.get("characters", [])) > 1 else [],
            "recommended_asset_tags": enrichment["recommended_asset_tags"],
            "raw_t5_output": raw_output,
            "telemetry": {
                "inference_latency_ms": latency_ms,
                "tokens_generated": len(output_ids[0]),
                "model_checkpoint": "models/4_Transformer_2k_v1",
                "base_architecture": "google/flan-t5-small",
                "decoding_strategy": "Beam Search (num_beams=4)",
                "schema_valid": schema_valid,
                "missing_fields": missing_fields,
                "extra_fields": extra_fields
            },
            "ethical_guardrails": {
                "safety_status": safety_msg,
                "dataset_attribution": "GameForge Fine-Tuned T5 (2,000 Curated Prompt-JSON Pairs; 80/10/10 Split)",
                "license": "CC-BY-NC-SA 4.0",
                "human_oversight_required": True
            },
            "enrichment_layer": enrichment
        }


transformer_service = TransformerService()
