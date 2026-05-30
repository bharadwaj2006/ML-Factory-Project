"""
=============================================================
LEXISCAN AUTO — STEP 3A: BERT FINE-TUNING (Transfer Learning)
=============================================================
PURPOSE:
    Fine-tune a pre-trained legal BERT model for NER.
    This gives much higher F1-Scores than the blank spaCy
    model in Step 2, because BERT already understands
    legal language from pre-training on legal corpora.

MODEL USED:
    'nlpaueb/legal-bert-base-uncased'
    Pre-trained on: EU legislation, contracts, court cases.
    Available at: https://huggingface.co/nlpaueb/legal-bert-base-uncased

LIBRARIES NEEDED:
    pip install transformers datasets torch seqeval

WHAT IS FINE-TUNING?
    BERT was pre-trained to understand legal language in general.
    Fine-tuning = we take those weights and train a small
    NER classification head on top, using OUR annotated data.
    Result: a model that speaks legalese AND knows our 4 entity types.

IOB2 TAGGING SCHEME:
    Each word (token) gets a tag:
        O              = not an entity
        B-DATE         = beginning of a DATE entity
        I-DATE         = inside a DATE entity (continuation)
        B-PARTY        = beginning of a PARTY entity
        I-PARTY        = inside a PARTY entity
        B-DOLLAR_AMOUNT
        I-DOLLAR_AMOUNT
        B-TERMINATION_CLAUSE
        I-TERMINATION_CLAUSE
=============================================================
"""

import torch
import numpy as np
import os
import json
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LABEL DEFINITIONS
# ─────────────────────────────────────────────
# IOB2 labels: O = Outside, B- = Beginning, I- = Inside
LABEL_LIST = [
    "O",
    "B-DATE",            "I-DATE",
    "B-PARTY",           "I-PARTY",
    "B-DOLLAR_AMOUNT",   "I-DOLLAR_AMOUNT",
    "B-TERMINATION_CLAUSE", "I-TERMINATION_CLAUSE",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ─────────────────────────────────────────────
# SAMPLE DATA IN IOB2 FORMAT
# ─────────────────────────────────────────────
# Each entry: list of (word, tag) tuples for one sentence.
# In production, load this from your Doccano/annotated corpus.

SAMPLE_IOB2_DATA = [
    [
        ("This", "O"), ("Agreement", "O"), ("is", "O"), ("dated", "O"),
        ("March", "B-DATE"), ("15", "I-DATE"), (",", "I-DATE"), ("2024", "I-DATE"),
        ("between", "O"),
        ("Meridian", "B-PARTY"), ("Capital", "I-PARTY"), ("LLC", "I-PARTY"),
        ("and", "O"),
        ("Vantage", "B-PARTY"), ("Holdings", "I-PARTY"), ("Inc.", "I-PARTY"),
    ],
    [
        ("The", "O"), ("fee", "O"), ("is", "O"),
        ("$2,500,000", "B-DOLLAR_AMOUNT"),
        ("payable", "O"), ("by", "O"),
        ("January", "B-DATE"), ("31", "I-DATE"), (",", "I-DATE"), ("2025", "I-DATE"),
    ],
    [
        ("Either", "B-TERMINATION_CLAUSE"), ("party", "I-TERMINATION_CLAUSE"),
        ("may", "I-TERMINATION_CLAUSE"), ("terminate", "I-TERMINATION_CLAUSE"),
        ("with", "I-TERMINATION_CLAUSE"), ("90", "I-TERMINATION_CLAUSE"),
        ("days", "I-TERMINATION_CLAUSE"), ("written", "I-TERMINATION_CLAUSE"),
        ("notice", "I-TERMINATION_CLAUSE"),
    ],
    [
        ("ABC", "B-PARTY"), ("Legal", "I-PARTY"), ("Partners", "I-PARTY"), ("LLP", "I-PARTY"),
        ("agrees", "O"), ("to", "O"), ("pay", "O"),
        ("$150,000", "B-DOLLAR_AMOUNT"),
        ("on", "O"),
        ("01/06/2024", "B-DATE"),
    ],
]


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class LegalNERDataset(Dataset):
    """
    PyTorch Dataset.
    Tokenizes word-level IOB2 data into BERT subword tokens.
    Handles the label alignment problem:
        "Meridian" → ["Mer", "##idian"] — only first subword gets the B- label
        continuation subwords get label -100 (ignored in loss calculation)
    """

    def __init__(self, iob2_data: list, tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = self._process(iob2_data)

    def _process(self, iob2_data):
        samples = []
        for sentence in iob2_data:
            words  = [w for w, _ in sentence]
            labels = [t for _, t in sentence]

            # Tokenize words → BERT subword tokens
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,     # words are pre-split
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            # Align labels to subword tokens
            word_ids    = encoding.word_ids()
            label_ids   = []
            prev_word   = None

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)    # special tokens [CLS], [SEP], [PAD]
                elif word_id != prev_word:
                    label_ids.append(LABEL2ID[labels[word_id]])   # first subword → real label
                else:
                    label_ids.append(-100)    # continuation subword → ignored
                prev_word = word_id

            samples.append({
                "input_ids":      encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels":         torch.tensor(label_ids, dtype=torch.long),
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────
# FINE-TUNE THE BERT MODEL
# ─────────────────────────────────────────────
def fine_tune_bert(
    iob2_data: list,
    model_output_dir: str,
    base_model: str = "nlpaueb/legal-bert-base-uncased",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
):
    """
    Fine-tune legal-BERT for NER on our 4 entity types.

    Args:
        iob2_data        : list of IOB2-tagged sentences
        model_output_dir : where to save the fine-tuned model
        base_model       : HuggingFace model ID (pre-trained)
        epochs           : training epochs (5 is typical for fine-tuning)
        batch_size       : keep small (4–8) if GPU memory is limited
        learning_rate    : 2e-5 is the standard BERT fine-tuning rate
    """

    # ── 1. Load tokenizer and model ──
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL_LIST),   # our 9 IOB2 labels
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    # ── 2. Prepare dataset ──
    # 80% train, 20% validation
    random_indices = list(range(len(iob2_data)))
    import random; random.shuffle(random_indices)
    split = int(len(iob2_data) * 0.8)

    train_dataset = LegalNERDataset([iob2_data[i] for i in random_indices[:split]],  tokenizer)
    val_dataset   = LegalNERDataset([iob2_data[i] for i in random_indices[split:]], tokenizer)
    logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # ── 3. Training arguments ──
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,           # L2 regularisation
        warmup_steps=50,             # gradual learning rate warm-up
        evaluation_strategy="epoch", # evaluate at end of every epoch
        save_strategy="epoch",
        load_best_model_at_end=True, # keep the best checkpoint
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none",            # disable WandB / TensorBoard for now
    )

    # ── 4. Compute F1 metrics during evaluation ──
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels = []
        pred_labels = []
        for pred_row, label_row in zip(predictions, labels):
            true_seq = []
            pred_seq = []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue   # skip special tokens
                true_seq.append(ID2LABEL[l])
                pred_seq.append(ID2LABEL[p])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq)

        # Simple F1 calculation (in production, use seqeval)
        tp = fp = fn = 0
        for true_seq, pred_seq in zip(true_labels, pred_labels):
            for t, p in zip(true_seq, pred_seq):
                if t != "O" and t == p: tp += 1
                elif t == "O" and p != "O": fp += 1
                elif t != "O" and p != t: fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    # ── 5. Data collator (handles dynamic padding) ──
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # ── 6. Create trainer and run ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # ── 7. Save model and tokenizer ──
    os.makedirs(model_output_dir, exist_ok=True)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"✓ Fine-tuned model saved to: {model_output_dir}")

    return model, tokenizer


# ─────────────────────────────────────────────
# PREDICT with fine-tuned BERT model
# ─────────────────────────────────────────────
def bert_predict(model_path: str, text: str) -> list:
    """
    Run NER prediction using the fine-tuned BERT model.
    Handles subword → word token alignment automatically.
    Returns list of {text, label, start, end} dicts.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    words = text.split()
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**encoding)

    logits   = outputs.logits[0]
    preds    = torch.argmax(logits, dim=-1).tolist()
    word_ids = encoding.word_ids()

    # Collect word-level predictions (skip subwords and special tokens)
    word_preds = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in word_preds:
            continue
        word_preds[word_id] = ID2LABEL[preds[idx]]

    # Group consecutive IOB2 tokens into entity spans
    entities = []
    current_entity = None
    char_pos = 0
    word_positions = []

    # Calculate character positions for each word
    pos = 0
    for word in words:
        word_positions.append((pos, pos + len(word)))
        pos += len(word) + 1  # +1 for space

    for word_id, label in word_preds.items():
        start_char, end_char = word_positions[word_id]

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text":  words[word_id],
                "label": label[2:],   # strip "B-"
                "start": start_char,
                "end":   end_char
            }
        elif label.startswith("I-") and current_entity:
            current_entity["text"] += " " + words[word_id]
            current_entity["end"]   = end_char
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_DIR = "models/bert_ner_model"

    # Train (comment out if model already trained)
    fine_tune_bert(
        iob2_data=SAMPLE_IOB2_DATA,
        model_output_dir=MODEL_DIR,
        epochs=5,
        batch_size=4,
    )

    # Predict on new text
    test_text = "On January 1 2025 Apex Financial Group and Summit Partners LLC agreed to a fee of $750,000."
    print(f"\nPrediction on:\n'{test_text}'\n")
    entities = bert_predict(MODEL_DIR, test_text)
    for e in entities:
        print(f"  [{e['label']}]  '{e['text']}'  (chars {e['start']}–{e['end']})")
