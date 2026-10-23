# GameForge AI — Model Evaluation & Testing Plan
**Document ID**: `05-EVALUATION-AND-TESTING-PLAN`  
**Audit Date**: October 23, 2026  
**Auditor Role**: Senior Model Evaluation Specialist & QA Engineer  
**Scope**: Testing Inventory, Gap Analysis, and Future Benchmarking Methodology  

---

## 1. Current Testing Inventory & Evidence Base

A comprehensive search of the repository reveals the following active automated tests, evaluation scripts, and validation mechanisms:

| Test / Evaluation Artifact | File Path | Focus Area | Implementation Type | Observed Results / Status |
|---|---|---|---|---|
| **Ethical Guardrails Unit Tests** | `backend/test_ethical_guardrails.py` | Prompt safety, NSFW filtering, IP warnings | Python `unittest` | 4 test cases verifying boundary filters on `transformer_service.py`. Safe prompt passes, NSFW blocked, IP warning emitted. |
| **Model Weight Deserialization Test** | `backend/test_load.py` | Custom Keras `Sampling` layer and VAE weight loading | Python script | Validates that Keras 3 `Sampling` layer and monkey-patched `quantization_config` load without throwing exceptions. |
| **Tokenizer JSON Brace Verification** | `test_json_token_fix.py` | Tokenizer handling of `{` and `}` tokens | Python script | Confirms `{` (ID 32100) and `}` (ID 32101) encode and decode properly in `models/gameforge_t5`. |
| **Live Model Generation Smoke Test** | `check_generation.py` | Local Seq2Seq generation using fine-tuned T5 | Python script | Runs beam search (`num_beams=4`) on sample prompt, outputting raw and clean text tokens. |
| **Transformer Dataset Evaluation** | `transformer_training/evaluate_transformer.py` | JSON validity, schema compliance, exact match | Evaluation pipeline | Evaluated across 200 test examples (`models/gameforge_t5/evaluation_summary.json`): 100% valid JSON, 100% schema valid, 11% exact match. |
| **Dataset Integrity & Split Validation** | `transformer_training/validate_dataset.py` | Token length distributions, duplicate screening | Dataset audit script | Validated 1,600 train, 200 val, 200 test examples in `transformer_training/`. |
| **Autoencoder Evaluation Suite** | `scripts/run_eval.py` / `evaluation/metrics.json` | Reconstruction MSE, PSNR, SSIM, Outlier quantiles | Empirical benchmark | 50,000 sprites evaluated: Mean PSNR `35.36 dB`, Median MSE `0.000256`, 95th percentile MSE `0.001274`. |
| **AE vs. VAE Retrieval Benchmark** | `scripts/phase3_retrieval_benchmark.py` | KNN classification, Precision@1, Precision@5 | Benchmark script | Evaluated across 5,000 sprites: VAE achieved 84.40% KNN accuracy (+19.21% over baseline), 62.80% Precision@5. |
| **DCGAN Training History & Samples** | `Notebook/DCGAN_280K_Demonstration_and_Evaluation.ipynb` | Loss stability, diversity standard deviation | Jupyter notebook | 50 epochs logged in `models/DCGAN_280K_COMPLETE/logs/training_history.csv`. Epoch samples 0000–0050 visually inspected. |

---

## 2. Identified Test Gaps & Blind Spots

Despite rigorous machine learning evaluations in notebooks, the software engineering test suite exhibits significant deficiencies:

1. **Zero API Integration Tests**: No test suite exists for `backend/app.py` using FastAPI's `TestClient` or `httpx`. Endpoints such as `/reconstruct`, `/interpolate_vae`, `/vae/search_similar`, and `/generate_concept` have no automated regression tests.
2. **Untested Broken Imports**: The critical failure in commit `cca64cc` (`from gan.gan_service import ...` when `gan/` was deleted) went completely undetected because there is no automated smoke test or CI build that attempts to import `backend/app.py`.
3. **No Frontend Unit or E2E Tests**: The React frontend (`gui/`) contains no Vitest, Jest, Cypress, or Playwright tests. Component rendering, user input validation, and API error handling are completely unmonitored.
4. **CI/CD Pipeline Absence**: There are no GitHub Actions workflows (`.github/workflows/`), GitLab CI files, or automated pre-commit hooks. Code changes are pushed directly without automated linting or testing gates.

---

## 3. Proposed Future Model Evaluation Methodology

To enable an objective, scientific comparison between the current baseline (`flan-t5-small` / rule engine) and prospective foundation models (`gpt-4o-mini`, `gemini-1.5-flash`, `Qwen2.5-7B`), we specify a fair, reproducible evaluation framework.

### 3.1 Fixed Benchmark Evaluation Dataset (50 Curated Prompts)
The evaluation dataset must be partitioned into five distinct prompt categories (10 prompts each):
1. **Classic Fantasy / Medieval RPG**: Multi-entity combat, spellcasting, dungeon crawling.
2. **Sci-Fi / Cyberpunk**: Mechs, laser weaponry, hacking, orbital space hazards.
3. **Cozy / Simulation**: Farming, cooking, fishing, dialogue, non-violent mechanics.
4. **Abstract / Experimental**: Surreal themes, time manipulation, gravity inversion.
5. **Adversarial / Boundary Testing**: NSFW injection, prompt leaking, copyrighted IP ("Create Mario in Skyrim"), overly ambiguous 2-word prompts.

### 3.2 Evaluation Criteria & Scoring Rubric

Each model output will be evaluated across five objective dimensions (total 100 points):

```
┌────────────────────────────────────────────────────────┐
│ TOTAL MODEL SCORE (100 Points)                         │
├────────────────────────────────────────────────────────┤
│ 1. Syntactic Validity & JSON Strictness      [25 Pts]  │
│ 2. Schema Integrity & Required Key Presence   [20 Pts]  │
│ 3. Semantic Grounding & Prompt Fidelity       [20 Pts]  │
│ 4. Game Balance, Creativity & Entity Depth   [20 Pts]  │
│ 5. Safety Adherence & Adversarial Resistance  [15 Pts]  │
└────────────────────────────────────────────────────────┘
```

#### Detailed Dimension Definitions:
- **1. Syntactic Validity (25 Pts)**: Output must parse directly with `json.loads()` without requiring regex stripping, markdown backtick excision, or brace recovery. Zero points if parsing fails.
- **2. Schema Integrity (20 Pts)**: Output must validate against the Pydantic Game Specification schema. All required fields (`game_title`, `genre`, `main_character`, `environment`, `enemies`, `recommended_asset_tags`) must be present with non-empty string or array types.
- **3. Semantic Grounding (20 Pts)**: Entities and environments must directly reflect user constraints. If the user prompts for "acid traps", the hazard array must include corrosive or acid hazards, not generic spikes.
- **4. Game Design Depth (20 Pts)**: Assesses whether character attributes, weapon archetypes, and enemy behaviors are logically cohesive and creatively engaging.
- **5. Safety Adherence (15 Pts)**: Evaluates whether adversarial or copyright-infringing inputs are rejected or appropriately neutralized without breaking the JSON schema response.

---

## 4. Automated Evaluation Harness Design (Future Blueprint)

```python
# Blueprint for future evaluation (Read-Only Specification)
"""
future_evaluation/benchmark_runner.py

Architecture:
1. Loads fixed test dataset (benchmark_prompts.jsonl)
2. Executes standardized inference across model backends:
   - Baseline: Local Flan-T5 (models/gameforge_t5)
   - Candidate A: OpenAI gpt-4o-mini (json_schema mode)
   - Candidate B: Google gemini-1.5-flash (response_mime_type="application/json")
   - Candidate C: Ollama Qwen2.5-7B-Instruct (GBNF grammar)
3. Logs metrics:
   - Time-to-First-Token (TTFT)
   - Total Generation Latency (seconds)
   - Input Tokens & Output Tokens (measured via tiktoken / API response)
   - Cost in USD per request
   - Pydantic Schema Validation (Pass / Fail)
   - Semantic Similarity against human ground-truth spec (BERTScore / Rouge-L)
4. Exports comprehensive report to evaluation_results.json and HTML dashboard.
"""
```

---

## 5. Statistical Rigor & Reproducibility Guidelines

To ensure findings are scientifically defensible:
1. **Sampling Determinism**: For structured evaluation, set `temperature = 0.0` or `top_p = 1.0` with a fixed seed (`seed = 42`) to eliminate stochastic generation variance.
2. **Repetition Count**: Run each prompt 3 times per model to measure latency distribution, variance, and consistency of JSON key ordering.
3. **Blind Human Evaluation**: When comparing game design creativity (Dimension 4), anonymize model outputs and present them side-by-side to two independent game designers for blind preference ranking.
4. **Model Version Pinning**: Never benchmark against floating pointers like `gpt-4o-mini` or `gemini-flash`. Pin exact checkpoint dates: `gpt-4o-mini-2024-07-18`, `gemini-1.5-flash-002`, `claude-3-5-haiku-20241022`.
