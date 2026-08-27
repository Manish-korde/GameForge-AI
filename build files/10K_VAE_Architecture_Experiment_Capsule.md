# GEN AI PROJECT — 10K VAE ARCHITECTURE EXPERIMENTATION CAPSULE

## 0. PURPOSE OF THIS CAPSULE

This capsule defines the next stage of the GEN AI project after completion of the 10K Autoencoder experiment and the large-scale 280K Autoencoder training.

The immediate goal is NOT to work on the 280K VAE.

For now, focus entirely on the controlled 10K experiment:

> Compare the existing 10K Autoencoder against a baseline 10K VAE and selected VAE variants under the same dataset conditions.

The purpose is to determine which VAE-family architecture is most suitable for the project's eventual large-scale model.

The 280K VAE decision/training is explicitly postponed.

---

# 1. PROJECT CONTEXT

The overall project is a Prompt-to-Game-Asset Generator.

The project includes multiple generative AI model families. For the current Unit 2 work, the relevant models are:

1. Autoencoder (AE)
2. Variational Autoencoder (VAE)
3. Potential VAE variants

The Autoencoder has already been implemented and trained.

The next major academic work is the VAE experimentation stage.

The project's academic minimum eventually requires two completed model implementations:

- Autoencoder
- VAE

Each should be trained on the intended large dataset and demonstrated through the GUI.

However, before committing to a VAE architecture for the large-scale model, we will perform a controlled 10K architecture experiment.

---

# 2. CURRENT STATE

## 2.1 Existing 10K Autoencoder

A complete 10K Autoencoder notebook already exists.

DO NOT unnecessarily recreate or replace it.

Existing AE notebook:

[PATH TO EXISTING 10K AE NOTEBOOK]

The existing notebook is the reference for:

- dataset loading
- preprocessing
- image size
- channels
- normalization
- deduplication
- split methodology
- random seed
- train/validation/test organization
- model configuration
- training methodology

Before implementing the VAE notebook, inspect the existing 10K AE notebook carefully.

The new VAE experiments must be compatible with its experimental setup.

---

# 3. IMPORTANT EXPERIMENTAL PRINCIPLE

The central purpose of the 10K experiment is a FAIR ARCHITECTURE COMPARISON.

Therefore:

## AE, VAE, and VAE variants must use the SAME dataset split.

Do NOT independently reshuffle the dataset for every model.

Do NOT generate a new train/validation/test split for the VAE.

Do NOT accidentally use a different validation set.

Do NOT compare models trained on different image populations.

The comparison should be:

    SAME DATA
       +
    SAME SPLIT
       +
    SAME PREPROCESSING
       +
    COMPARABLE TRAINING CONDITIONS
       ↓
    ARCHITECTURE DIFFERENCE
       ↓
    FAIR COMPARISON

This is one of the most important requirements of the experiment.

---

# 4. EXACT SPLIT REQUIREMENT

The existing 10K Autoencoder experiment already has a defined dataset split.

The VAE notebook MUST reuse the exact same split.

If the existing AE split is not shuffled dynamically during training, preserve the same underlying train/validation/test assignment.

If the AE experiment generated explicit indices, use those indices.

Prefer saving and reusing:

- train indices
- validation indices
- test indices

rather than regenerating them.

The VAE notebook should verify:

1. Number of training images
2. Number of validation images
3. Number of test images
4. No overlap between splits
5. Same image IDs/indices as the AE experiment

If the AE notebook has a fixed seed and deterministic split procedure, reproduce it exactly only if the original indices are unavailable.

Do NOT silently create a different split.

---

# 5. WHY THE SAME VALIDATION SET MATTERS

The validation set must remain identical across experiments.

For example:

    AE
       ↓
    Validation Set V

    VAE
       ↓
    Validation Set V

    VAE Variant
       ↓
    Validation Set V

This means validation loss and model-selection behavior can be compared meaningfully.

The same principle applies to the test set.

The final test evaluation should use the exact same test images for all models.

---

# 6. NEW NOTEBOOK PLAN

Create a new notebook dedicated to the 10K VAE experiments.

Recommended filename:

    10K_VAE_Architecture_Comparison.ipynb

Recommended location:

    Notebook/
        10K_VAE_Architecture_Comparison.ipynb

The existing 10K Autoencoder notebook remains separate.

Do NOT merge the new VAE implementation into the existing AE notebook unless there is a strong technical reason.

---

# 7. WHAT THE NEW NOTEBOOK WILL CONTAIN

The new notebook should be a reusable experiment framework rather than a one-off script.

It should support:

    Model 1 → VAE
    Model 2 → selected VAE variant
    Model 3 → optional additional VAE variant

The notebook should allow one model to be trained/evaluated at a time through configuration.

Example concept:

    MODEL_TO_RUN = "VAE"

or:

    MODEL_TO_RUN = "VQ_VAE"

The exact implementation mechanism can be chosen based on the existing notebook style.

The important requirement is that each experiment can be run independently and saved independently.

---

# 8. BASELINE MODEL: VAE

The first new model must be a conventional Variational Autoencoder.

Conceptual architecture:

    Input Image
        ↓
    Encoder
        ↓
    μ
    log(σ²)
        ↓
    Reparameterization
        ↓
    z
        ↓
    Decoder
        ↓
    Reconstructed Image

The VAE differs from the conventional AE because the encoder learns a latent probability distribution rather than only a deterministic latent vector.

The model should implement the reparameterization trick:

    z = μ + σ × ε

where:

    ε ~ N(0, I)

---

# 9. VAE LOSS

The baseline VAE must use the standard two-part objective:

    Total VAE Loss
        =
    Reconstruction Loss
        +
    KL Divergence

Conceptually:

    L = L_reconstruction + β L_KL

For the first baseline experiment, use a straightforward standard configuration unless the existing project methodology specifies otherwise.

A reasonable baseline is:

    β = 1

The implementation must separately track:

- reconstruction loss
- KL loss
- total loss
- validation total loss

Do not report only one combined number without explaining its components.

---

# 10. ARCHITECTURE FAIRNESS

The VAE should be architecturally comparable to the existing 10K AE.

Do not make the VAE dramatically larger merely because it is a new model.

Where practical, keep comparable:

- input size
- output size
- encoder capacity
- decoder capacity
- latent dimension
- batch size
- optimizer
- learning-rate strategy
- preprocessing
- training split

The main architectural difference should be the probabilistic latent representation and VAE objective.

This makes the AE vs VAE comparison scientifically more defensible.

---

# 11. VAE VARIANT EXPERIMENTS

After the baseline VAE works, test one or more carefully selected VAE variants.

Potential candidates include:

- VQ-VAE
- β-VAE
- another appropriate VAE-family architecture

Do NOT automatically implement every VAE variant.

Do NOT add variants merely to increase the number of models.

The selected variant must have a clear reason for being relevant to the Prompt-to-Game-Asset Generator.

---

# 12. VQ-VAE IS NOT AUTOMATICALLY REQUIRED

VQ-VAE is currently a candidate, not a guaranteed final choice.

It is substantially different from a conventional VAE because it introduces vector quantization and a discrete latent codebook.

Therefore, before implementing it, verify that:

1. It can be trained reliably on the 10K dataset.
2. It can be evaluated meaningfully against the baseline VAE.
3. It provides a useful advantage for this project's game-asset generation objective.
4. It can realistically be demonstrated before the project deadline.

If another VAE variant provides a clearer academic and practical benefit, use that instead.

Do not force VQ-VAE into the project simply because it sounds more advanced.

---

# 13. RECOMMENDED EXPERIMENT ORDER

Run the experiments in this order:

## Experiment 1

Existing:

    10K Autoencoder

This is the baseline.

DO NOT retrain it if its existing split, preprocessing, architecture, and results already satisfy the controlled experiment requirements.

## Experiment 2

New:

    10K baseline VAE

Use the exact same split.

## Experiment 3

New:

    10K selected VAE variant

Use the exact same split.

## Optional Experiment 4

A second VAE variant only if there is enough time and a clear research reason.

---

# 14. DO NOT TRAIN EVERYTHING BLINDLY IN ONE SESSION

Cloud notebook sessions may have a limited runtime, such as approximately 10 hours.

Do NOT design the project around the assumption that all models must finish in one session.

Instead, make every model experiment independently resumable.

Preferred workflow:

    Session 1
        ↓
    Train VAE
        ↓
    Save checkpoints
        ↓
    Save best model
        ↓
    Save history
        ↓
    Evaluate

    Session 2
        ↓
    Train VAE variant
        ↓
    Save checkpoints
        ↓
    Save best model
        ↓
    Evaluate

The exact number of sessions is not important.

The saved artifacts are what matter.

---

# 15. CHECKPOINTING REQUIREMENT

Every new VAE experiment must support interruption/restart.

The notebook should save:

- latest checkpoint
- best checkpoint/model
- training history
- experiment configuration
- epoch/state information

If the runtime disconnects, the experiment should resume rather than restart from epoch 1.

Do NOT rely on the notebook session itself to preserve training.

Save important artifacts to persistent storage.

---

# 16. MODEL ARTIFACT ORGANIZATION

Use separate directories for each model.

Suggested structure:

    models/
        10K/
            AE/
                best.keras
                latest.keras

            VAE/
                best.keras
                latest.keras

            VAE_variant/
                best.keras
                latest.keras

Do not overwrite one model with another.

---

# 17. EXPERIMENT ARTIFACT ORGANIZATION

Suggested:

    experiments/
        10K/
            AE/
                config/
                checkpoints/
                logs/
                evaluation/
                samples/

            VAE/
                config/
                checkpoints/
                logs/
                evaluation/
                samples/

            VAE_variant/
                config/
                checkpoints/
                logs/
                evaluation/
                samples/

This allows each experiment to remain independently reproducible.

---

# 18. REQUIRED EVALUATION FOR EACH MODEL

Every model tested in the 10K comparison should have the same core evaluation.

At minimum:

## Reconstruction

Calculate:

- Test MSE
- optionally MAE
- optionally PSNR

Use the SAME test set.

## Visual reconstruction

Display:

    Original → Reconstruction

using representative test images.

Use the same test examples where possible so visual comparison is direct.

## Training curves

Plot:

- training loss
- validation loss

For VAE, also plot:

- reconstruction loss
- KL loss

## Generation

For VAE-family models, generate new samples from random latent vectors.

For example:

    z ~ N(0, I)
        ↓
    decoder
        ↓
    generated game asset

## Latent-space analysis

For VAE models, include useful latent-space demonstrations.

At minimum:

- random sampling
- latent interpolation if practical

---

# 19. AE VS VAE COMPARISON

The final comparison should not be based solely on MSE.

The project is a generative game-asset project.

Compare:

### Reconstruction

- MSE
- visual fidelity
- PSNR if useful

### Generation

- quality of random samples
- diversity
- plausibility as game assets

### Latent representation

- smoothness
- interpolation behavior
- meaningful variation

### Training behavior

- convergence
- stability
- reconstruction/KL balance

### Complexity

- parameter count
- training time
- inference time if practical

---

# 20. IMPORTANT INTERPRETATION RULE

Do NOT assume:

    lower MSE = universally better model

For an AE, reconstruction is its primary purpose.

For a VAE, the objective also includes learning a structured generative latent space.

Therefore, a VAE may have worse pixel-level reconstruction than an AE while still being more useful for:

- sampling
- latent interpolation
- controlled variation
- generative asset creation

The final comparison must acknowledge this.

---

# 21. FINAL 10K EXPERIMENT QUESTION

The experiment should answer:

> Under the same 10K dataset, preprocessing, train/validation/test split, and comparable training conditions, which architecture provides the best balance of reconstruction quality, latent-space behavior, generative quality, and practical complexity for the Prompt-to-Game-Asset Generator?

The answer determines which VAE architecture moves forward.

---

# 22. SELECTION OF THE FINAL VAE ARCHITECTURE

After completing the 10K experiments:

    AE
     \
      \
       → comparison → selected architecture
      /
    VAE
      \
       VAE variant

The selected VAE architecture will later become the candidate for the 280K experiment.

IMPORTANT:

The 280K VAE training is OUT OF SCOPE for this current stage.

Do not begin it automatically after the 10K experiments.

First complete and analyze the 10K comparison.

---

# 23. 280K WORK IS FROZEN FOR NOW

The following work is intentionally postponed:

- 280K VAE architecture selection for production
- 280K VAE training
- additional 280K VAE variants
- final AE vs VAE large-scale comparison

The existing 280K AE remains safely completed and available as the large-scale AE baseline.

The current work is ONLY:

    10K AE
       vs
    10K VAE
       vs
    10K VAE variant(s)

---

# 24. CURRENT HIGH-LEVEL PROJECT ROADMAP

    PHASE 1 — 10K CONTROLLED EXPERIMENT
            │
            ├── Existing 10K AE
            │
            ├── 10K VAE
            │
            └── 10K VAE variant
                    │
                    ▼
              ARCHITECTURE
               COMPARISON
                    │
                    ▼
              SELECT VAE FAMILY
                    │
                    │
                    X  280K VAE postponed
                    │
                    ▼
              NEXT PROJECT STAGE

The existing 280K Autoencoder is already complete and should not be disturbed.

---

# 25. IMPORTANT DO-NOT-DO LIST

DO NOT:

- create a new random split for VAE
- use a different validation set
- use a different test set
- change preprocessing between models
- overwrite the AE model
- retrain the existing 10K AE unnecessarily
- immediately jump to 280K VAE
- implement many VAE variants without justification
- assume VQ-VAE is automatically the best
- compare only training loss
- compare only reconstruction MSE
- fabricate missing metrics
- require all models to finish in one 10-hour session
- lose checkpoints because the runtime disconnected

---

# 26. SUCCESS CRITERIA FOR THIS STAGE

This stage is complete when:

[ ] Existing 10K AE experiment is identified as the baseline.

[ ] Exact AE train/validation/test split is preserved.

[ ] New 10K VAE notebook is implemented.

[ ] VAE uses the exact same split.

[ ] VAE training can resume after interruption.

[ ] VAE best model is saved.

[ ] VAE reconstruction is evaluated.

[ ] VAE random generation is demonstrated.

[ ] VAE latent interpolation is demonstrated if feasible.

[ ] At least one justified VAE variant is tested.

[ ] Variant uses the same dataset split.

[ ] AE vs VAE vs variant comparison is produced.

[ ] Results are saved in reproducible form.

[ ] A VAE architecture is selected for possible future 280K training.

[ ] 280K VAE training remains postponed until explicitly started.

---

# 27. FINAL EXPERIMENTAL STRUCTURE

The project should now be understood as:

                    10K CONTROLLED STUDY
                           │
             ┌─────────────┼─────────────┐
             │             │             │
            AE            VAE       VAE VARIANT
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                    FAIR COMPARISON
                           │
                           ▼
                 SELECT BEST VAE DESIGN
                           │
                           ▼
                 FUTURE 280K TRAINING

The purpose of the 10K stage is not to replace the 280K dataset.

The purpose is to select and justify the architecture before committing expensive compute to large-scale training.

---

# 28. IMMEDIATE NEXT ACTION

Do NOT start coding yet.

First inspect the existing 10K Autoencoder notebook and identify exactly:

1. Dataset source
2. Dataset loading method
3. Deduplication method
4. Image dimensions
5. Number of channels
6. Normalization
7. Train count
8. Validation count
9. Test count
10. Exact split/index generation
11. Random seed
12. Latent dimension
13. Encoder architecture
14. Decoder architecture
15. Batch size
16. Optimizer
17. Learning rate
18. Loss
19. Epoch configuration
20. Existing saved model/results

Then use those values as the fixed experimental baseline for the new VAE notebook.

Do not guess any of these values if they can be obtained from the existing notebook.

---

# 29. CORE PRINCIPLE

The 10K experiment is a controlled scientific experiment.

The 280K model is the eventual scale-up.

Therefore:

    10K = ARCHITECTURE SELECTION
    280K = LARGE-SCALE MODEL TRAINING

For the current stage:

    10K AE
    10K VAE
    10K VAE variant(s)

with the SAME split, SAME validation set, SAME test set, and comparable conditions.

The 280K VAE is deliberately postponed.
