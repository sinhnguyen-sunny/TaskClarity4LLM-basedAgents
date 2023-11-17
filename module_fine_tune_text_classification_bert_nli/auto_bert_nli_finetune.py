# Import Necessary Libraries
import pandas as pd
import numpy as np
import torch
import datasets
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict, Dataset
import optuna
import argparse

# Argument Parsing for Hyperparameters


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for NLI")
    parser.add_argument("--dataset_path", type=str,
                        required=True, help="Path to the NLI dataset")
    parser.add_argument("--model_name", type=str,
                        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", help="Model name or path")
    parser.add_argument("--max_length", type=int,
                        default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training and evaluation batch size")
    parser.add_argument("--numer_of_trials", type=int, default=5,
                        help="Increasing this value can lead to better hyperparameters, but will take longer")
    parser.add_argument("--seed", type=int, default=2023,
                        help="Seed Gloabal parameters")

    # Add other arguments as needed
    return parser.parse_args()


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

# Utility Function: Clean Memory


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# Load and Preprocess Data


def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    label_mapping = {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}
    df['label'] = df['Label'].replace(label_mapping)
    return df

# Model and Tokenizer Initialization


def model_init():
    clean_memory()
    return AutoModelForSequenceClassification.from_pretrained(args.model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Compute Metrics Function


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

# Main Function


def main():
    args = parse_args()

    # global variables
    SEED_GLOBAL = args.seed
    numer_of_trials = args.numer_of_trials

    # Data Preprocessing
    label_mapping = {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, model_max_length=512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_and_prepare_data(args.dataset_path, label_mapping)
    dataset = tokenize_dataset(dataset, tokenizer)
    # remove unnecessary columns for model training
    dataset = dataset.remove_columns([
        'Task', '__index_level_0__', 'Label'])
    dataset = datasets.load_dataset(
        'csv', data_files={'train': args.dataset_path})  # Update as needed
    dataset_hp = dataset["train"].train_test_split(
        test_size=0.4, seed=2023, shuffle=True)

    label_text_alphabetical = np.sort(dataset['train'].column_names)

    # Training Arguments
    train_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer
    auto_trainer = Trainer(
        model_init=model_init,
        args=train_args,
        train_dataset=dataset_hp["train"],
        eval_dataset=dataset_hp["test"],
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred, label_text_alphabetical)
    )

    # Hyperparameter Tuning

    # Define the hyperparameter space
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "seed": trial.suggest_int("seed", 1, 40),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        }

    # Optuna sampler configuration
    optuna_sampler = optuna.samplers.TPESampler(
        seed=SEED_GLOBAL,
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
        n_startup_trials=numer_of_trials / 2,
        n_ei_candidates=24,
        multivariate=False,
        group=False,
        warn_independent_sampling=True,
        constant_liar=False
    )

    # Hyperparameter search using Trainer
    best_run = auto_trainer.hyperparameter_search(
        n_trials=numer_of_trials,
        direction="maximize",
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics["eval_f1_macro"],
        backend='optuna',
        sampler=optuna_sampler
    )

    # Display best hyperparameters
    print("Best trial:", best_run)

    # Update the training arguments with the best hyperparameters
    for key, value in best_run.hyperparameters.items():
        setattr(train_args, key, value)

    # Reinitialize the Trainer with the best hyperparameters
    auto_trainer = Trainer(
        model_init=model_init,
        args=train_args,
        train_dataset=dataset_hp["train"],
        eval_dataset=dataset_hp["test"],
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred, label_text_alphabetical)
    )

    # Training with the best hyperparameters
    auto_trainer.train()

    # Evaluate the model
    results = auto_trainer.evaluate()
    print(f'Automatically Fine-tuned Results:\n {results}')


if __name__ == "__main__":
    main()
