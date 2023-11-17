
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict, Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
import argparse
import gc


def load_and_prepare_data(dataset_path, label_mapping):
    df = pd.read_csv(dataset_path)
    print(f"Original Dataset: {df.describe()}")
    df['label'] = df['Label'].replace(label_mapping)
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    return DatasetDict({
        "train": Dataset.from_pandas(train),
        "test": Dataset.from_pandas(test)
    })


def tokenize_dataset(dataset, tokenizer, max_length=512):
    def tokenize_nli_format(examples):
        return tokenizer(examples["Premise"], examples["Hypothesis"], truncation=True, max_length=max_length)
    return dataset.map(tokenize_nli_format, batched=True)


def compute_metrics(eval_pred, label_text_alphabetical):
    predictions, labels = eval_pred

    # Function to yield successive n-sized chunks from lst
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Reformat model output to enable calculation of standard metrics
    softmax = torch.nn.Softmax(dim=1)
    prediction_chunks_lst = list(
        chunks(predictions, len(label_text_alphabetical)))
    hypo_position_highest_prob = [
        np.argmax(chunk[:, 0]) for chunk in prediction_chunks_lst]

    label_chunks_lst = list(chunks(labels, len(label_text_alphabetical)))
    label_position_gold = [np.argmin(chunk) for chunk in label_chunks_lst]

    # Calculate standard metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        label_position_gold, hypo_position_highest_prob, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        label_position_gold, hypo_position_highest_prob, average='micro')
    acc_balanced = balanced_accuracy_score(
        label_position_gold, hypo_position_highest_prob)
    acc_not_balanced = accuracy_score(
        label_position_gold, hypo_position_highest_prob)

    metrics = {
        'accuracy': acc_not_balanced,
        'f1_macro': f1_macro,
        'accuracy_balanced': acc_balanced,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
    }

    return metrics

# Utility Function: Clean Memory


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def run_manual_fine_tune(dataset_path, model_name, train_args):
    label_mapping = {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, model_max_length=512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_and_prepare_data(dataset_path, label_mapping)
    dataset = tokenize_dataset(dataset, tokenizer)
    # remove unnecessary columns for model training
    dataset = dataset.remove_columns([
        'Task', '__index_level_0__', 'Label'])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name).to(device)
    label_text_alphabetical = np.sort(dataset['train'].column_names)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred, label_text_alphabetical=label_text_alphabetical)
    )
    trainer.train()
    results = trainer.evaluate()
    print(f'Manual Fine-tuned Results:{results}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Manual fine-tuned Bert-NLI training')
    parser.add_argument("--dataset_path", type=str,
                        required=True, help="Path to the NLI dataset")
    parser.add_argument("--model_name", type=str,
                        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", help="Model name for fine-tuning")
    parser.add_argument("--max_length", type=int,
                        default=512, help="Max token length")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--logging_dir", type=str,
                        default="./logs", help="Logging directory")
    parser.add_argument("--learning_rate", type=float,
                        default=2e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int,
                        default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int,
                        default=80, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float,
                        default=0.25, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float,
                        default=0.1, help="Weight decay")

    args = parser.parse_args()
    # Create TrainingArguments instance
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        report_to=[]
    )

    # Clean memory
    clean_memory()

    # Call run_manual_fine_tune with separated arguments
    run_manual_fine_tune(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        train_args=train_args
    )
