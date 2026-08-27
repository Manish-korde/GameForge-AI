# GEN AI Project — 280K Autoencoder Experiment
## Full Chat / Project Continuity Capsule

**Last updated:** 2026-08-19  
**Current status:** Final 280K training run has just started. Estimated remaining runtime: ~7 hours.

---

## 1. Purpose

This document is the continuity capsule for the current GEN AI project. It records the project goal, dataset, preprocessing, model architecture, benchmarks, training configuration, first training run, checkpoint failure, corrected checkpoint strategy, and the exact current state.

**Critical rule:** Do not change the validated architecture, dataset split, batch size, learning rate, or training objective without a clear technical reason.

---

## 2. Project Goal

The project is an image autoencoder experiment.

The original experiment used approximately 10K images. The objective is to scale the same validated autoencoder architecture to a much larger, deduplicated dataset of approximately 280K images.

Dataset:

`evilsocket/alucard-sprites`

Current experiment name:

`AE_280K_full`

The cleaned dataset contains **282,511 unique images**.

The project intentionally keeps the exact 10K architecture instead of automatically making the network larger merely because the dataset is larger.

---

## 3. Previous 10K Baseline

Known previous result:

- Clean Test MSE: approximately **0.001251**

This is the baseline for later comparison.

Important: after the new run, compare **10K test MSE vs 280K test MSE**. Do not directly compare the 10K test MSE to the first run's 280K validation MSE because they are different splits.

---

## 4. Dataset

Hugging Face dataset:

`evilsocket/alucard-sprites`

Original dataset:

```text
312,550 images
features: image, text
```

Example image:

```text
128 × 128
RGBA
array shape: (128, 128, 4)
uint8
pixel range: 0–255
```

Example caption:

`pixel art, gray, small, wizard, mage, spellcaster, back view`

---

## 5. Deduplication

Exact deduplication was completed.

```text
Source images   : 312,550
Unique images   : 282,511
Exact duplicates: 30,039
```

Therefore:

```text
312,550 - 30,039 = 282,511 unique images
```

Do not accidentally revert to the raw 312,550-image dataset.

---

## 6. Deterministic Shuffle

Seed:

`42`

The unique images were deterministically shuffled and the ordering was saved.

Important files:

```text
/kaggle/working/AE_280K/config/unique_indices.json
/kaggle/working/AE_280K/config/shuffled_unique_indices.json
```

---

## 7. Train / Validation / Test Split

Total unique images:

`282,511`

Split:

```text
Training   : 254,260
Validation : 14,126
Test       : 14,125
```

Percentages:

```text
Training   : 90.0000%
Validation : 5.0002%
Test       : 4.9998%
```

Verified:

- no train/validation/test overlap
- every unique image assigned exactly once

Saved manifests:

```text
/kaggle/working/AE_280K/config/train_indices.json
/kaggle/working/AE_280K/config/val_indices.json
/kaggle/working/AE_280K/config/test_indices.json
/kaggle/working/AE_280K/config/dataset_manifest.json
```

These dataset artifacts survived the later runtime restart.

---

## 8. Environment

Original validated environment:

```text
TensorFlow : 2.20.0
Python     : 3.12.13
Platform   : Linux-6.12.90+-x86_64-with-glibc2.35
GPU count  : 2
```

Actual GPUs:

```text
GPU 0: Tesla T4
GPU 1: Tesla T4
```

TensorFlow used:

```text
MirroredStrategy
```

Distributed communication:

```text
NCCL
num_devices = 2
group_size = 2
```

Therefore the actual training uses both GPUs.

---

## 9. Input Pipeline

Images are loaded lazily.

Input/target:

```text
shape : (128, 128, 4)
dtype : float32
range : [0, 1]
```

Pipeline checks confirmed:

- dataset objects exist
- image loads successfully
- correct shape
- float32 dtype
- normalized range [0, 1]
- lazy pipeline works
- batching and prefetching work

---

## 10. Autoencoder Architecture

The architecture is exactly the validated 10K architecture.

Model:

```text
encoder (Sequential)
Output: (None, 8, 8, 256)
Params: 388,704

decoder (Sequential)
Output: (None, 128, 128, 4)
Params: 690,404
```

Total:

```text
1,079,108 parameters
```

All parameters are trainable.

The model is created inside:

```python
strategy.scope()
```

using the two-GPU `MirroredStrategy`.

---

## 11. Training Configuration

Locked configuration:

```text
experiment_name       : AE_280K_full
dataset_name          : evilsocket/alucard-sprites
image_height          : 128
image_width           : 128
channels              : 4
dataset_size_target   : 280000
train_ratio           : 0.90
validation_ratio      : 0.05
test_ratio            : 0.05
batch_size_per_gpu    : 32
global_batch_size     : 64
epochs                : 50
learning_rate         : 0.001
optimizer             : Adam
loss                  : MSE
seed                  : 42
checkpoint_every_epoch: True
```

---

## 12. Batch Benchmark

Candidate per-GPU batch sizes:

```text
8
16
32
```

Corresponding global batches:

```text
8/GPU  → 16
16/GPU → 32
32/GPU → 64
```

Benchmark:

```text
per-GPU  global  images/sec
8        16      132.32
16       32      432.30
32       64      543.32
```

Batch 32/GPU was selected.

---

## 13. Real Training-Step Benchmark

A real backpropagation benchmark was performed:

```text
Per-GPU batch : 32
Global batch  : 64
Steps         : 10
```

Result:

```text
Processed images       : 640
Elapsed time            : 1.02 sec
Seconds per step       : 0.1016
Training images/sec    : 630.06
Average benchmark loss : 0.198868
Optimizer iterations   : 0 → 12
```

Confirmed:

- forward pass
- backpropagation
- Adam optimizer updates
- batch 32/GPU fits
- two-GPU distributed training

---

## 14. Benchmark Reset

Because the benchmark changed optimizer state, the model was rebuilt before the actual experiment.

Fresh-model verification:

```text
Optimizer iterations: 0
Total parameters    : 1,079,108
Output shape        : (None, 128, 128, 4)
```

This ensured benchmark updates were not carried into real training.

---

## 15. First 280K Training Run

The first 280K training run genuinely completed all 50 epochs.

Typical speed:

```text
~90–94 ms/step
3972 steps/epoch
```

Final:

```text
Last completed epoch    : 50
Final optimizer iterations: 198650
```

Best result:

```text
Best validation loss : 0.0004909808631055057
Best epoch           : 32
```

Selected validation losses:

```text
Epoch 1  : 0.00392591
Epoch 2  : 0.00248531
Epoch 3  : 0.00201780
Epoch 4  : 0.00124372
Epoch 5  : 0.00110990
Epoch 6  : 0.00100444
Epoch 7  : 0.00099972
Epoch 9  : 0.00087390
Epoch 10 : 0.00083391
Epoch 13 : 0.00075717
Epoch 14 : 0.00066907
Epoch 18 : 0.00062105
Epoch 19 : 0.00060294
Epoch 21 : 0.00058798
Epoch 22 : 0.00056816
Epoch 24 : 0.00052905
Epoch 27 : 0.00051629
Epoch 30 : 0.00050730
Epoch 32 : 0.00049098  ← BEST
```

After epoch 32, validation performance fluctuated:

```text
Epoch 34 : 0.00090217
Epoch 35 : 0.00054124
Epoch 37 : 0.00049282
Epoch 40 : 0.00060310
Epoch 45 : 0.00068545
Epoch 49 : 0.00074966
Epoch 50 : 0.00056520
```

Therefore the best checkpoint should be preferred over simply taking epoch 50.

---

## 16. Why the First Run Was Lost

After reopening the Kaggle notebook, Python's live variables were gone:

```python
history
autoencoder
CONFIG
training_state
```

The notebook preserves code, not live Python memory.

The `/kaggle/working/AE_280K` directory itself still existed, but:

```text
checkpoints/
  latest/
  best/
```

were empty.

A search across:

```text
/kaggle/working
/kaggle/input
/tmp
```

found:

```text
NO .keras / .h5 / .hdf5 MODEL FILES FOUND.
```

The original `training_state.json` was also overwritten when setup cells were rerun, so it currently no longer contains the original epoch-50 state.

The dataset manifests and configuration survived, but the actual first-run neural-network weights are not available in the current runtime.

This means the first 50-epoch computation must be reproduced.

---

## 17. Checkpoint Design Lesson

The previous callback printed:

```text
[CHECKPOINT] Latest saved after epoch N
```

but a print statement is not proof that an artifact will survive a Kaggle runtime lifecycle.

The corrected system now:

- saves `.keras`
- prints the actual file size
- creates explicit final model files
- verifies files exist
- asserts nonzero file size
- saves best and final models separately

This must not be removed.

---

## 18. Corrected Cell 19

Cell 19 now initializes:

```text
LATEST_MODEL_PATH
BEST_MODEL_PATH
TRAINING_STATE_PATH
HISTORY_PATH
CONFIG_PATH
```

Latest checkpoint:

```text
/kaggle/working/AE_280K/checkpoints/latest/latest.keras
```

Best checkpoint:

```text
/kaggle/working/AE_280K/checkpoints/best/best.keras
```

The state is initialized for a fresh run.

---

## 19. Corrected Cell 21

Cell 21 now has three callbacks:

```text
TrainingStateCallback
LatestCheckpointCallback
BestCheckpointCallback
```

Every epoch:

```text
latest.keras is saved
training_state.json is updated
an easy-to-find latest export copy is created
actual checkpoint file size is printed
```

When validation loss improves:

```text
best.keras is saved
best export copy is created
best_val_loss is updated
best_epoch is updated
actual best file size is printed
```

Easy-to-find copies:

```text
/kaggle/working/AE_280K/AE_280K_latest.keras
/kaggle/working/AE_280K/AE_280K_best.keras
```

---

## 20. Corrected Cell 22

Cell 22 is now the final training cell.

It:

1. builds the train and validation datasets
2. uses the clean model from Cell 20
3. confirms optimizer iteration is 0
4. creates fresh checkpoint callbacks
5. runs 50 epochs
6. saves the final epoch-50 model
7. saves the encoder
8. copies the best checkpoint into the models directory
9. saves training history
10. updates training state
11. verifies all required artifacts exist
12. prints actual file sizes

Final planned artifacts:

```text
/kaggle/working/AE_280K/models/AE_280K_final.keras
/kaggle/working/AE_280K/models/AE_280K_best.keras
/kaggle/working/AE_280K/models/AE_280K_encoder.keras
/kaggle/working/AE_280K/logs/history.json
/kaggle/working/AE_280K/logs/training_state.json
```

The most important file is:

```text
AE_280K_best.keras
```

because it represents the best validation model.

---

## 21. CURRENT STATE — VERY IMPORTANT

### The second 280K training run has JUST STARTED.

Current status:

```text
TRAINING IN PROGRESS
Estimated remaining runtime: ~7 hours
```

It is using the corrected Cell 22.

Current locked setup:

```text
2 × Tesla T4
MirroredStrategy
NCCL
32 images/GPU
64 global batch
Adam
learning rate 0.001
MSE
50 epochs
```

The architecture is unchanged:

```text
1,079,108 parameters
```

The dataset is unchanged:

```text
282,511 unique images
```

Split:

```text
254,260 train
14,126 validation
14,125 test
```

### DO NOT:

- rerun Cell 22 while it is already training
- restart the runtime unnecessarily
- change the architecture
- change batch size
- change learning rate
- change dataset
- stop the training merely because it is taking hours

---

## 22. What To Do When Training Finishes

Do NOT immediately close the Kaggle session.

First wait for:

```text
TRAINING COMPLETE
```

and the artifact verification.

We need to see nonzero sizes for:

```text
AE_280K_final.keras
AE_280K_best.keras
AE_280K_encoder.keras
history.json
training_state.json
```

The final output should contain:

```text
Completed epochs: 50
Best epoch: XX
Best val_loss: XXXXXXXXXX
Final optimizer iterations: XXXXX
```

Then verify/persist/download the model artifacts before closing or restarting the Kaggle runtime.

Order:

```text
TRAINING FINISHES
        ↓
VERIFY FILES
        ↓
PERSIST / DOWNLOAD
        ↓
ONLY THEN EVALUATE
```

---

## 23. Evaluation After Training

Use the untouched test set:

```text
14,125 images
```

Evaluation should include:

### Quantitative

- Test MSE
- reconstruction-error statistics
- comparison against 10K baseline
- best epoch vs final epoch

### Qualitative

Create side-by-side:

```text
Original | Reconstruction
```

for multiple test images.

### Training analysis

Generate:

- training-loss curve
- validation-loss curve
- overfitting analysis
- best epoch identification

### Artifacts

Keep:

```text
full autoencoder
best model
encoder
training history
configuration
dataset manifests
evaluation metrics
reconstruction samples
plots
```

---

## 24. Important Comparison Rule

Known 10K result:

```text
Clean Test MSE ≈ 0.001251
```

First 280K result:

```text
Best Validation MSE ≈ 0.00049098
Best epoch = 32
```

These are different metrics/splits.

The proper final comparison is:

```text
10K Test MSE
vs
280K Test MSE
```

on the appropriate test data.

Do not claim a definitive 280K improvement from the validation-vs-test comparison alone.

---

## 25. Intended Final Directory

```text
AE_280K/
│
├── checkpoints/
│   ├── latest/
│   │   └── latest.keras
│   └── best/
│       └── best.keras
│
├── models/
│   ├── AE_280K_final.keras
│   ├── AE_280K_best.keras
│   └── AE_280K_encoder.keras
│
├── logs/
│   ├── training_state.json
│   ├── history.json
│   ├── training.log
│   └── training_log.csv
│
├── config/
│   ├── experiment_config.json
│   ├── dataset_manifest.json
│   ├── unique_indices.json
│   ├── shuffled_unique_indices.json
│   ├── train_indices.json
│   ├── val_indices.json
│   └── test_indices.json
│
├── evaluation/
│   ├── metrics
│   ├── plots
│   └── reconstruction samples
│
└── samples/
```

---

## 26. Critical Numbers

```text
Dataset source:
evilsocket/alucard-sprites

Original images:
312,550

Exact duplicates:
30,039

Unique images:
282,511

Train:
254,260

Validation:
14,126

Test:
14,125

Seed:
42

Image:
128 × 128 × 4 RGBA

Architecture:
1,079,108 parameters

Encoder output:
8 × 8 × 256

GPUs:
2 × Tesla T4

Strategy:
MirroredStrategy

Per-GPU batch:
32

Global batch:
64

Learning rate:
0.001

Optimizer:
Adam

Loss:
MSE

Epochs:
50

10K clean test MSE:
~0.001251

First 280K best validation MSE:
0.0004909808631055057

First 280K best epoch:
32

First 280K completed epochs:
50

First 280K final optimizer iterations:
198650

Current second 280K run:
IN PROGRESS

Estimated remaining:
~7 hours
```

---

## 27. Future Chat Continuity

If this capsule is loaded in a new chat, understand:

**This project is not starting from scratch.**

Dataset selection, deduplication, deterministic split, architecture, GPU setup, batch-size benchmark, learning rate, optimizer, loss, and 50-epoch training plan are already decided.

The first 50-epoch run completed successfully and reached:

```text
best validation MSE = 0.00049098
best epoch = 32
```

but its actual trained weights were not retained after the runtime restart.

The current second run is reproducing that experiment with corrected artifact-saving logic.

### Current task:

> Let the current 50-epoch 280K training run finish, verify the actual `.keras` artifacts, persist/download them, and only then proceed to test evaluation.

**Do not accidentally start a third training run.**

---
