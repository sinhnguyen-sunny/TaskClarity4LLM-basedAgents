import unittest
import pandas as pd
import torch
from manual_bert_nli_finetune import load_and_prepare_data, tokenize_dataset, compute_metrics
from transformers import AutoTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Unit tests for manual_bert_nli_finetune.py')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for tokenization')
    return parser.parse_args()

class TestManualBertNLIFinetune(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        args = parse_args()
        cls.dataset_path = args.dataset_path
        cls.model_name = args.model_name
        cls.label_mapping = {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}

    def test_load_and_prepare_data(self):
        df = pd.read_csv(self.dataset_path)
        self.assertTrue(len(df) > 0, "Dataset is empty")

        dataset = load_and_prepare_data(self.dataset_path, self.label_mapping)
        self.assertIn('train', dataset, "Training set missing in dataset")
        self.assertIn('test', dataset, "Test set missing in dataset")

    def test_tokenize_dataset(self):
        dataset = load_and_prepare_data(self.dataset_path, self.label_mapping)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        tokenized_dataset = tokenize_dataset(dataset['train'], tokenizer)
        self.assertIsNotNone(tokenized_dataset, "Tokenization failed")

    def test_compute_metrics(self):
        predictions = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))
        label_text_alphabetical = ['Entailment', 'Neutral', 'Contradiction']
        metrics = compute_metrics((predictions, labels), label_text_alphabetical)
        expected_keys = ['accuracy', 'f1_macro', 'accuracy_balanced', 'f1_micro', 'precision_macro', 'recall_macro', 'precision_micro', 'recall_micro']
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")

if __name__ == '__main__':
    unittest.main()
