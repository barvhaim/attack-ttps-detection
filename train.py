import logging
import pandas as pd
import numpy as np
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.preprocessing import MultiLabelBinarizer as MLB
from sklearn.model_selection import train_test_split
from datasets import Dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the classes (ATT&CK techniques) the model is trained on
CLASSES = [
    "T1003.001",
    "T1005",
    "T1012",
    "T1016",
    "T1021.001",
    "T1027",
    "T1033",
    "T1036.005",
    "T1041",
    "T1047",
    "T1053.005",
    "T1055",
    "T1056.001",
    "T1057",
    "T1059.003",
    "T1068",
    "T1070.004",
    "T1071.001",
    "T1072",
    "T1074.001",
    "T1078",
    "T1082",
    "T1083",
    "T1090",
    "T1095",
    "T1105",
    "T1106",
    "T1110",
    "T1112",
    "T1113",
    "T1140",
    "T1190",
    "T1204.002",
    "T1210",
    "T1218.011",
    "T1219",
    "T1484.001",
    "T1518.001",
    "T1543.003",
    "T1547.001",
    "T1548.002",
    "T1552.001",
    "T1557.001",
    "T1562.001",
    "T1564.001",
    "T1566.001",
    "T1569.002",
    "T1570",
    "T1573.001",
    "T1574.002",
]

# Initialize the MultiLabelBinarizer with the expected classes
mlb = MLB(classes=CLASSES)
mlb.fit([[c] for c in CLASSES])

tokenizer = BertTokenizer.from_pretrained(
    "allenai/scibert_scivocab_uncased", max_length=512
)
model = BertForSequenceClassification.from_pretrained("scibert_multi_label_model")


def _convert_to_dataset(df):
    # Transform labels to binary format
    label_matrix = mlb.transform(df["labels"].tolist()).astype(np.float32)

    # Create dataset dictionary
    dataset_dict = {
        "sentence": df["sentence"].tolist(),
        "labels": label_matrix.tolist(),
    }

    return Dataset.from_dict(dataset_dict)


def _tokenize_function(examples):
    tokenized_examples = tokenizer(
        examples["sentence"], padding="max_length", truncation=True, max_length=512
    )
    tokenized_examples["labels"] = examples["labels"]
    return tokenized_examples


def _compute_metrics(pred):
    # Convert logits to predictions (0 or 1)
    predictions = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).int().numpy()

    # Convert label_ids to integers (they might be floats)
    labels = pred.label_ids.astype(np.int32)

    # Manual calculation of metrics for multi-label classification
    # True Positives: sum of predictions * labels (element-wise)
    tp = np.sum(predictions * labels)

    # All Predicted Positives
    pred_pos = np.sum(predictions)

    # All Actual Positives
    actual_pos = np.sum(labels)

    # Calculate precision, recall, F1
    precision = tp / pred_pos if pred_pos > 0 else 0
    recall = tp / actual_pos if actual_pos > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Calculate per-class F1 for macro averaging
    num_classes = predictions.shape[1]
    class_f1_scores = []

    for i in range(num_classes):
        class_preds = predictions[:, i]
        class_labels = labels[:, i]

        class_tp = np.sum(class_preds * class_labels)
        class_pred_pos = np.sum(class_preds)
        class_actual_pos = np.sum(class_labels)

        class_precision = class_tp / class_pred_pos if class_pred_pos > 0 else 0
        class_recall = class_tp / class_actual_pos if class_actual_pos > 0 else 0

        if (class_precision + class_recall) > 0:
            class_f1 = (
                2 * class_precision * class_recall / (class_precision + class_recall)
            )
        else:
            class_f1 = 0

        class_f1_scores.append(class_f1)

    # Macro F1 is the average of per-class F1 scores
    f1_macro = np.mean(class_f1_scores) if class_f1_scores else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,  # This is the micro F1
        "f1_macro": f1_macro,
    }


def train():
    input_df = pd.read_json("multi_label.json").drop(columns="doc_title").head(500)
    train_df, eval_df = train_test_split(
        input_df, test_size=0.2, random_state=42, shuffle=True
    )

    logger.info(f"Train size: {train_df.shape[0]}")
    logger.info(f"Eval size: {eval_df.shape[0]}")

    logger.info("Converting to dataset...")
    train_dataset = _convert_to_dataset(train_df)
    eval_dataset = _convert_to_dataset(eval_df)

    logger.info("Tokenizing datasets...")

    train_dataset = train_dataset.map(_tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(_tokenize_function, batched=True)

    logger.info("Training...")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,  # Total number of training epochs
        per_device_train_batch_size=10,  # Batch size for training
        per_device_eval_batch_size=20,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=100,  # Log every X updates steps
        eval_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",  # Save checkpoint every epoch
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="f1",  # Use F1 score to determine the best model
        save_total_limit=2,  # Only keep the 2 best checkpoints
        report_to="none",  # Disable reporting to wandb/tensorboard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ],  # Stop training if no improvement for 3 evaluations
    )

    trainer.train()

    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")

    # Save the best model
    model_save_path = "./fine_tuned_scibert_multi_label"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Test the model
    predictions = trainer.predict(eval_dataset)
    preds = (torch.sigmoid(torch.tensor(predictions.predictions)) > 0.5).int().numpy()

    # Convert predictions back to label format
    pred_labels = mlb.inverse_transform(preds)

    # Get the actual labels directly from the evaluation dataset
    actual_labels = mlb.inverse_transform(np.array(eval_dataset["labels"]))

    results = pd.DataFrame(
        {
            "sentence": eval_df["sentence"].tolist(),
            "predicted": [frozenset(p) for p in pred_labels],
            "actual": [frozenset(a) for a in actual_labels],
        }
    )

    # Calculate true positives, false positives, and false negatives
    tp = (
        results.apply((lambda r: r.predicted & r.actual), axis=1)
        .explode()
        .value_counts()
    )
    fp = (
        results.apply((lambda r: r.predicted - r.actual), axis=1)
        .explode()
        .value_counts()
    )
    fn = (
        results.apply((lambda r: r.actual - r.predicted), axis=1)
        .explode()
        .value_counts()
    )

    # Combine into a single DataFrame
    counts = pd.concat({"tp": tp, "fp": fp, "fn": fn}, axis=1).fillna(0).astype(int)

    # Calculate support (number of occurrences of each class)
    support = (
        pd.Series([t for a in actual_labels for t in a]).value_counts().rename("#")
    )

    # Calculate precision, recall, and F1 for each class
    p = counts.tp.div(counts.tp + counts.fp).fillna(0)
    r = counts.tp.div(counts.tp + counts.fn).fillna(0)
    f1 = (2 * p * r) / (p + r).replace(0, float("nan")).fillna(0)

    # Combine scores
    scores = (
        pd.concat({"P": p, "R": r, "F1": f1}, axis=1)
        .fillna(0)
        .sort_values(by="F1", ascending=False)
    )

    # Calculate macro scores
    scores.loc["(macro)"] = scores.mean()

    # Calculate micro scores
    micro = counts.sum()
    scores.loc["(micro)", "P"] = mP = micro.tp / (micro.tp + micro.fp)
    scores.loc["(micro)", "R"] = mR = micro.tp / (micro.tp + micro.fn)
    scores.loc["(micro)", "F1"] = (2 * mP * mR) / (mP + mR)

    # Display detailed performance metrics
    logger.info("\nDetailed Performance Metrics:")
    result_table = scores.join(support)
    logger.info(result_table)


if __name__ == "__main__":
    train()
