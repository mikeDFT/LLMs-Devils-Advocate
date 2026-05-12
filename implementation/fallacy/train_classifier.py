"""
Fine-tune ModernBERT-base on the logical-fallacy dataset for 13-class fallacy classification.

Output: implementation/fallacy_model/
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, ClassLabel
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FALLACY_BASE_MODEL, FALLACY_DATASET, FALLACY_MODEL_DIR, DATA_DIR

FALLACY_NUM_LABELS = 13
FALLACY_LABELS = [
    "ad_hominem",
    "ad_populum",
    "appeal_to_emotion",
    "circular_reasoning",
    "equivocation",
    "fallacy_of_credibility",
    "fallacy_of_extension",
    "fallacy_of_logic",
    "fallacy_of_relevance",
    "false_causality",
    "false_dilemma",
    "intentional",
    "faulty_generalization",
]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    # macro F1 prevents the model from ignoring rare fallacies
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "f1": f1}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def main():
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be slow.")

    print(f"Loading dataset: {FALLACY_DATASET}")
    ds = load_dataset(FALLACY_DATASET)

    label_names = sorted(set(ds["train"]["logical_fallacies"]))
    print(f"Found {len(label_names)} unique labels in dataset: {label_names}")

    label_to_id = {name: i for i, name in enumerate(FALLACY_LABELS)}
    id_to_label = {i: name for i, name in enumerate(FALLACY_LABELS)}

    def encode_labels(example):
        raw_label = str(example["logical_fallacies"])
        normalized_label = raw_label.replace(" ", "_") # fallacies look like "appeal to emotion" in the dataset, we need "appeal_to_emotion"
        example["label_id"] = label_to_id.get(normalized_label, 0)
        return example

    ds = ds.map(encode_labels)
    ds = ds.cast_column("label_id", ClassLabel(names=FALLACY_LABELS))

    if "test" not in ds:
        ds = ds["train"].train_test_split(test_size=0.15, seed=42, stratify_by_column="label_id")

    train_label_ids = np.array(ds["train"]["label_id"])
    class_weights = compute_class_weight("balanced", classes=np.arange(FALLACY_NUM_LABELS), y=train_label_ids)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class weights (balanced): min={class_weights.min():.3f} max={class_weights.max():.3f}")

    print(f"Loading tokenizer and model: {FALLACY_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FALLACY_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FALLACY_BASE_MODEL,
        num_labels=FALLACY_NUM_LABELS,
        id2label=id_to_label,
        label2id=label_to_id,
        classifier_dropout=0.1,
    )

    def tokenize(example):
        return tokenizer(
            example["source_article"],
            truncation=True,
            max_length=512,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label_id", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # init dynamic collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(FALLACY_MODEL_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting ModernBERT training...")
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Final eval accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Final eval Macro F1: {eval_results['eval_f1']:.4f}")

    # Detailed classification report
    preds_output = trainer.predict(ds["test"])
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    
    print("\nClassification Report:")
    report_labels = [id_to_label[i] for i in range(FALLACY_NUM_LABELS)]
    print(classification_report(labels, preds, target_names=report_labels, labels=range(FALLACY_NUM_LABELS), zero_division=0))

    trainer.save_model(str(FALLACY_MODEL_DIR))
    print(f"Model saved to {FALLACY_MODEL_DIR}")


if __name__ == "__main__":
    main()
