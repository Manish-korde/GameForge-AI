# GameForge AI: Responsible AI & Ethical Deployment Guidelines

## Overview & Scope
GameForge AI is an open-source, non-commercial game asset generation framework designed to empower indie developers while upholding rigorous Responsible AI and ethical deployment standards.

This document establishes the official governance rules across four core deployment pillars:
1. **Content Safety & Prompt Boundary Filtering**
2. **Dataset Creator Attribution & Licensing Transparency**
3. **Human-in-the-Loop Assistive Co-Pilot Oversight**
4. **Non-Commercial Research Boundaries & IP Protection**

---

## Pillar 1: Content Safety & Prompt Boundary Filtering

### Prompt Audit Engine (`backend/transformer_service.py`)
All natural language inputs submitted via the Transformer Semantic Planner (`/generate_concept`) pass through a real-time Prompt Safety Engine before invoking model synthesis or vector search.

* **Prohibited Content Filtering**: Real-time regex boundaries intercept explicit, abusive, or harmful categories (NSFW, violence/gore, hate speech, self-harm).
  * **Violation Response**: Returns an HTTP 400 response with explicit safety metadata (`detail: "Prompt contains restricted or unsafe content ('<term>')"`).
* **Trademark & IP Re-Grounding**: Mention of third-party commercial gaming intellectual property (e.g., *Mario*, *Pokemon*, *Zelda*) generates a warning log and automatically re-grounds requests into generic retro archetypes to prevent trademark infringement.

---

## Pillar 2: Dataset Creator Attribution & Licensing

### Dataset Manifest & Compliance
GameForge AI synthesizes assets and fine-tunes models strictly using open-access, non-commercial datasets. Every API payload returned by `/generate_concept` or `/vae/search_similar` embeds explicit attribution metadata.

| Dataset Name | Population | Primary License | System Role |
| :--- | :--- | :--- | :--- |
| **`evilsocket/alucard-sprites`** | 282,511 RGBA Sprites | `CC-BY-NC-SA 4.0` | 32-bit Sprite VAE/AE Reconstruction & Vector Similarity Search |
| **`GEM/viggo`** | 6,900 Dialogue Samples | `CC-BY 4.0` | Game Spec Intent & Dialogue Transformer Parsing |
| **`Fraser/pico-8-games`** | 10,967 Cartridges | `CC-BY-NC-SA 4.0` | Retro 8x8 Tilemaps, PICO-8 Palette & Cartridge Mechanics |

### Automated Attribution Tags
Every similarity search result or generated concept spec includes `ethical_guardrails` metadata:
```json
"ethical_guardrails": {
  "content_safety": "Passed Safety Audit",
  "license": "CC-BY-NC-SA 4.0",
  "dataset_source": "evilsocket/alucard-sprites",
  "attribution_required": true,
  "usage_boundary": "Non-Commercial Research & Prototyping Only"
}
```

---

## Pillar 3: Human-in-the-Loop Assistive Co-Pilot

GameForge AI is explicitly designed as a **co-pilot** rather than an autonomous generator that replaces creative artists.

* **Designer Approval Checkpoint**: The GUI (`ConceptPlanner.jsx`) enforces explicit human review. Generative asset pipelines cannot be triggered until a designer checks the *Human Designer Oversight* approval checkbox.
* **Interactive Latent Exploration**: Artists control linear interpolation ($\alpha \in [0, 1]$) and latent vector search sliders, retaining full creative direction over style and character aesthetics.

---

## Pillar 4: Non-Commercial Research & IP Protection

* **Commercial Restriction**: Assets generated or reconstructed from `CC-BY-NC-SA 4.0` source datasets must not be commercialized without express permission from dataset creators.
* **Open Science & Transparency**: Benchmark metrics ($R^2 = 0.9997$ for reconstruction, $P@1 = 0.66$ for retrieval) and source code are open to enable peer verification and academic research.

---

## API Status Endpoint
The backend exposes live governance metadata via the `/ethical_guidelines` GET endpoint, ensuring frontend clients and external tools can inspect compliance parameters dynamically.
