"""
GameForge AI - Fine-tune FLAN-T5-small

Natural-language game request
            ->
Structured JSON game specification

This version fixes the JSON brace problem discovered during inference.

The original T5 tokenizer maps:
    { -> <unk>
    } -> <unk>

Therefore this training script adds { and } as regular tokenizer
tokens before fine-tuning.

It does NOT modify:
    - app.py
    - AE
    - VAE
    - dataset contents

Run from project root:

    python transformer_training\train_transformer.py
"""

import inspect
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np


# ===========================================================================
# IMPORTS
# ===========================================================================

try:
    import torch

    from datasets import load_dataset

    from transformers import (
        AutoConfig,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

except ImportError as e:
    print("ERROR: A required package is missing.")
    print(f"Details: {e}")

    print("\nMake sure you:")
    print("  1. Activated:")
    print("       .venv_transformer\\Scripts\\activate")

    print("  2. Installed:")
    print("       python -m pip install -r requirements_transformer.txt")

    sys.exit(1)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

SEED = 42

MODEL_NAME = "google/flan-t5-small"

TASK_PREFIX = "game concept: "

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 512

NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

WARMUP_STEPS = 100

LOGGING_STEPS = 25

MAX_GRAD_NORM = 1.0

USE_FP16 = False


# ===========================================================================
# PATHS
# ===========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

TRAIN_FILE = SCRIPT_DIR / "train.jsonl"

VALIDATION_FILE = SCRIPT_DIR / "validation.jsonl"

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "gameforge_t5_checkpoints"
)

FINAL_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "gameforge_t5"
)


# ===========================================================================
# SEEDS
# ===========================================================================

def set_all_seeds(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# ===========================================================================
# FILE CHECK
# ===========================================================================

def check_files_exist():

    missing = []

    if not TRAIN_FILE.exists():
        missing.append(str(TRAIN_FILE))

    if not VALIDATION_FILE.exists():
        missing.append(str(VALIDATION_FILE))

    if missing:

        print("\nERROR: Required dataset files not found:")

        for path in missing:
            print(f"  - {path}")

        print(
            "\nExpected:"
            "\n  transformer_training\\train.jsonl"
            "\n  transformer_training\\validation.jsonl"
        )

        sys.exit(1)


# ===========================================================================
# DEVICE
# ===========================================================================

def detect_device_config():

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        total_mem_gb = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        if total_mem_gb >= 6:
            batch_size = 8
        else:
            batch_size = 4

        print(
            f"GPU detected: {gpu_name} "
            f"({total_mem_gb:.1f} GB VRAM)"
        )

        print(
            "FP16 mixed precision: DISABLED"
        )

        print(
            f"Batch size: {batch_size}"
        )

        return "cuda", False, batch_size

    print("No CUDA GPU detected.")

    print("Training will run on CPU.")

    return "cpu", False, 2


# ===========================================================================
# TOKENIZER
# ===========================================================================

def build_tokenizer():

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(
        f"Original tokenizer vocabulary: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"Original tokenizer length: "
        f"{len(tokenizer)}"
    )

    print(
        f"Original {repr('{')} token: "
        f"{tokenizer.convert_tokens_to_ids('{')}"
    )

    print(
        f"Original {repr('}')} token: "
        f"{tokenizer.convert_tokens_to_ids('}')}"
    )

    # ------------------------------------------------------------------
    # IMPORTANT
    #
    # Do NOT use additional_special_tokens here.
    #
    # They would be removed by:
    #
    #     skip_special_tokens=True
    #
    # during inference.
    #
    # We want { and } to be ordinary tokens.
    # ------------------------------------------------------------------

    tokens_to_add = []

    if tokenizer.convert_tokens_to_ids("{") == tokenizer.unk_token_id:
        tokens_to_add.append("{")

    if tokenizer.convert_tokens_to_ids("}") == tokenizer.unk_token_id:
        tokens_to_add.append("}")

    if tokens_to_add:

        added = tokenizer.add_tokens(
            tokens_to_add
        )

        print(
            f"\nAdded {added} JSON structure tokens:"
        )

        for token in tokens_to_add:
            print(
                f"  {repr(token)}"
            )

    else:

        print(
            "\nJSON braces already exist in tokenizer."
        )

    print(
        f"Final tokenizer length: "
        f"{len(tokenizer)}"
    )

    print(
        f"New {repr('{')} token id: "
        f"{tokenizer.convert_tokens_to_ids('{')}"
    )

    print(
        f"New {repr('}')} token id: "
        f"{tokenizer.convert_tokens_to_ids('}')}"
    )

    return tokenizer


# ===========================================================================
# MODEL
# ===========================================================================

def build_model(tokenizer, device):

    print(
        f"\nLoading model: {MODEL_NAME}"
    )

    # ------------------------------------------------------------------
    # Load config first.
    #
    # The previous checkpoint warning showed that the saved model
    # contained separate shared/lm_head weights while config requested
    # tied weights.
    #
    # We explicitly make the configuration consistent.
    # ------------------------------------------------------------------

    config = AutoConfig.from_pretrained(
        MODEL_NAME
    )

    config.tie_word_embeddings = False

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        config=config,
    )

    # ------------------------------------------------------------------
    # Add room for the new { and } tokenizer tokens.
    # ------------------------------------------------------------------

    old_size = model.get_input_embeddings().num_embeddings

    new_size = len(tokenizer)

    if new_size != old_size:

        print(
            f"\nResizing model embeddings:"
        )

        print(
            f"  Old size: {old_size}"
        )

        print(
            f"  New size: {new_size}"
        )

        model.resize_token_embeddings(
            new_size
        )

    else:

        print(
            f"\nEmbedding size already matches: "
            f"{new_size}"
        )

    # ------------------------------------------------------------------
    # Keep configuration consistent.
    # ------------------------------------------------------------------

    model.config.tie_word_embeddings = False

    model.config.use_cache = False

    model = model.to(device)

    return model


# ===========================================================================
# DATA PREPROCESSING
# ===========================================================================

def preprocess_function(
    examples,
    tokenizer,
):

    inputs = [
        TASK_PREFIX + text
        for text in examples["input"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# ===========================================================================
# TOKENIZER VALIDATION
# ===========================================================================

def validate_json_tokens(tokenizer):

    print("\n" + "=" * 70)
    print("TOKENIZER JSON CHECK")
    print("=" * 70)

    test_json = (
        '{"theme":"Survival",'
        '"environment":"Dense Wilderness",'
        '"characters":[{"name":"Stranded Survivor",'
        '"desc":"A lone survivor."}]}'
    )

    ids = tokenizer.encode(
        test_json,
        add_special_tokens=False,
    )

    tokens = tokenizer.convert_ids_to_tokens(
        ids
    )

    unk_count = tokens.count(
        tokenizer.unk_token
    )

    print(
        f"Encoded tokens: {len(tokens)}"
    )

    print(
        f"UNK tokens: {unk_count}"
    )

    print(
        f"'{{' token id: "
        f"{tokenizer.convert_tokens_to_ids('{')}"
    )

    print(
        f"'}}' token id: "
        f"{tokenizer.convert_tokens_to_ids('}')}"
    )

    if unk_count != 0:

        print(
            "\n❌ TOKENIZER CHECK FAILED"
        )

        print(
            "JSON still contains <unk> tokens."
        )

        print(
            "Training will NOT start."
        )

        sys.exit(1)

    print(
        "\n✅ TOKENIZER CHECK PASSED"
    )

    print(
        "JSON braces are represented by real tokens."
    )

    print("=" * 70)


# ===========================================================================
# TRAINING ARGUMENTS
# ===========================================================================

def build_training_arguments(
    output_dir,
    use_fp16,
    batch_size,
):

    sig = inspect.signature(
        Seq2SeqTrainingArguments.__init__
    )

    if "eval_strategy" in sig.parameters:

        strategy_kwarg = "eval_strategy"

    else:

        strategy_kwarg = "evaluation_strategy"

    kwargs = {

        "output_dir": str(output_dir),

        "num_train_epochs": NUM_EPOCHS,

        "per_device_train_batch_size": batch_size,

        "per_device_eval_batch_size": batch_size,

        "gradient_accumulation_steps": 1,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "warmup_steps": WARMUP_STEPS,

        "max_grad_norm": MAX_GRAD_NORM,

        "optim": "adamw_torch",

        "logging_steps": LOGGING_STEPS,

        "logging_nan_inf_filter": False,

        "save_strategy": "epoch",

        "save_total_limit": 3,

        "load_best_model_at_end": True,

        "metric_for_best_model": "eval_loss",

        "greater_is_better": False,

        "predict_with_generate": True,

        "fp16": use_fp16,

        "bf16": False,

        "report_to": "none",

        "seed": SEED,

        strategy_kwarg: "epoch",
    }

    return Seq2SeqTrainingArguments(
        **kwargs
    )


# ===========================================================================
# SANITY CHECK
# ===========================================================================

def run_pretraining_sanity_check(
    model,
    data_collator,
    dataset,
    device,
):

    print("\n" + "=" * 70)
    print("PRE-TRAINING SANITY CHECK")
    print("=" * 70)

    examples = dataset.select(
        range(min(2, len(dataset)))
    )

    features = []

    for i in range(len(examples)):

        features.append(
            {
                "input_ids":
                    examples[i]["input_ids"],

                "attention_mask":
                    examples[i]["attention_mask"],

                "labels":
                    examples[i]["labels"],
            }
        )

    batch = data_collator(
        features
    )

    print(
        f"Input shape : "
        f"{tuple(batch['input_ids'].shape)}"
    )

    print(
        f"Label shape : "
        f"{tuple(batch['labels'].shape)}"
    )

    labels = batch["labels"]

    ignored_count = (
        labels == -100
    ).sum().item()

    non_ignored_count = (
        labels != -100
    ).sum().item()

    print(
        f"Valid label tokens   : "
        f"{non_ignored_count}"
    )

    print(
        f"Ignored label tokens : "
        f"{ignored_count}"
    )

    batch = {
        key: value.to(device)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }

    model.eval()

    with torch.no_grad():

        outputs = model(
            **batch
        )

    loss = outputs.loss.item()

    print(
        f"Forward-pass loss : "
        f"{loss:.6f}"
    )

    print(
        f"Loss finite       : "
        f"{math.isfinite(loss)}"
    )

    if non_ignored_count == 0:

        print(
            "\n❌ SANITY CHECK FAILED"
        )

        print(
            "All labels are ignored."
        )

        sys.exit(1)

    if not math.isfinite(loss):

        print(
            "\n❌ SANITY CHECK FAILED"
        )

        print(
            "Loss is NaN/Inf."
        )

        sys.exit(1)

    print(
        "\n✅ PRE-TRAINING SANITY CHECK PASSED"
    )

    print(
        "Dataset → Tokenizer → Collator → Model → Loss is healthy."
    )

    print("=" * 70)


# ===========================================================================
# MAIN
# ===========================================================================

def main():

    print("=" * 70)
    print("GameForge AI - FLAN-T5 JSON-FORMAT FIX TRAINING")
    print("=" * 70)

    check_files_exist()

    set_all_seeds(SEED)

    device, use_fp16, batch_size = (
        detect_device_config()
    )

    # ------------------------------------------------------------------
    # TOKENIZER
    # ------------------------------------------------------------------

    tokenizer = build_tokenizer()

    validate_json_tokens(
        tokenizer
    )

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------

    model = build_model(
        tokenizer,
        device,
    )

    print(
        f"\nModel vocabulary size: "
        f"{model.get_input_embeddings().num_embeddings}"
    )

    print(
        f"Tokenizer length: "
        f"{len(tokenizer)}"
    )

    print(
        f"tie_word_embeddings: "
        f"{model.config.tie_word_embeddings}"
    )

    # ------------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------------

    print("\nLoading dataset...")

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALIDATION_FILE),
        },
    )

    print(
        f"Train examples      : "
        f"{len(raw_datasets['train'])}"
    )

    print(
        f"Validation examples : "
        f"{len(raw_datasets['validation'])}"
    )

    # ------------------------------------------------------------------
    # TOKENIZATION
    # ------------------------------------------------------------------

    print(
        "\nTokenizing dataset..."
    )

    tokenized_datasets = raw_datasets.map(
        lambda batch: preprocess_function(
            batch,
            tokenizer,
        ),
        batched=True,
        remove_columns=raw_datasets[
            "train"
        ].column_names,
    )

    # ------------------------------------------------------------------
    # COLLATOR
    # ------------------------------------------------------------------

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    # ------------------------------------------------------------------
    # SANITY CHECK
    # ------------------------------------------------------------------

    run_pretraining_sanity_check(
        model=model,
        data_collator=data_collator,
        dataset=tokenized_datasets["train"],
        device=device,
    )

    # ------------------------------------------------------------------
    # TRAINING ARGS
    # ------------------------------------------------------------------

    training_args = build_training_arguments(
        CHECKPOINT_DIR,
        use_fp16,
        batch_size,
    )

    # ------------------------------------------------------------------
    # TRAINER
    # ------------------------------------------------------------------

    trainer = Seq2SeqTrainer(
        model=model,

        args=training_args,

        train_dataset=tokenized_datasets[
            "train"
        ],

        eval_dataset=tokenized_datasets[
            "validation"
        ],

        data_collator=data_collator,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3
            )
        ],
    )

    # ------------------------------------------------------------------
    # TRAINING INFORMATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)

    print("STARTING TRAINING")

    print("=" * 70)

    print(
        f"Model          : {MODEL_NAME}"
    )

    print(
        f"Train examples : "
        f"{len(tokenized_datasets['train'])}"
    )

    print(
        f"Validation     : "
        f"{len(tokenized_datasets['validation'])}"
    )

    print(
        f"Epochs         : {NUM_EPOCHS}"
    )

    print(
        f"Learning rate  : {LEARNING_RATE}"
    )

    print(
        f"Batch size     : {batch_size}"
    )

    print(
        f"FP16           : {use_fp16}"
    )

    print(
        f"Optimizer      : adamw_torch"
    )

    print(
        f"Warmup steps   : {WARMUP_STEPS}"
    )

    print(
        f"Grad clipping  : {MAX_GRAD_NORM}"
    )

    print(
        "\nJSON brace tokens:"
    )

    print(
        f"  {{ -> "
        f"{tokenizer.convert_tokens_to_ids('{')}"
    )

    print(
        f"  }} -> "
        f"{tokenizer.convert_tokens_to_ids('}')}"
    )

    print("=" * 70)

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    try:

        train_result = trainer.train()

    except RuntimeError as e:

        print("\n" + "=" * 70)

        print("❌ TRAINING FAILED")

        print("=" * 70)

        print(str(e))

        raise

    # ------------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)

    print("TRAINING FINISHED")

    print("=" * 70)

    final_train_loss = (
        train_result.training_loss
    )

    print(
        f"Final train loss: "
        f"{final_train_loss:.6f}"
    )

    if not math.isfinite(
        final_train_loss
    ):

        print(
            "\n❌ Final training loss is NaN/Inf."
        )

        sys.exit(1)

    # ------------------------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------------------------

    FINAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nSaving best model to:"
    )

    print(
        f"  {FINAL_MODEL_DIR}"
    )

    trainer.save_model(
        str(FINAL_MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(FINAL_MODEL_DIR)
    )

    # ------------------------------------------------------------------
    # SAVE METRICS
    # ------------------------------------------------------------------

    metrics_file = (
        FINAL_MODEL_DIR
        / "training_metrics.json"
    )

    metrics = {

        "model_name":
            MODEL_NAME,

        "train_examples":
            len(tokenized_datasets["train"]),

        "validation_examples":
            len(tokenized_datasets["validation"]),

        "epochs_requested":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            batch_size,

        "fp16":
            use_fp16,

        "optimizer":
            "adamw_torch",

        "warmup_steps":
            WARMUP_STEPS,

        "max_grad_norm":
            MAX_GRAD_NORM,

        "seed":
            SEED,

        "tie_word_embeddings":
            False,

        "json_brace_tokens_added":
            True,

        "open_brace_token_id":
            tokenizer.convert_tokens_to_ids("{"),

        "close_brace_token_id":
            tokenizer.convert_tokens_to_ids("}"),

        "final_train_loss":
            final_train_loss,
    }

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        f"\nTraining metrics saved:"
    )

    print(
        f"  {metrics_file}"
    )

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)

    print("✅ TRAINING COMPLETE")

    print("=" * 70)

    print(
        "\nFinal model:"
    )

    print(
        f"  {FINAL_MODEL_DIR}"
    )

    print(
        "\nThe AE, VAE, and app.py were NOT modified."
    )

    print(
        "\nNext:"
    )

    print(
        "  1. Check tokenizer"
    )

    print(
        "  2. Test one generated JSON"
    )

    print(
        "  3. Run evaluate_transformer.py"
    )

    print(
        "  4. Only then integrate into app.py"
    )


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()