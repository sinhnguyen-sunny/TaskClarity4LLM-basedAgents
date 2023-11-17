# Manual Fine-Tuning of BERT-NLI Models

## Overview

This repository contains scripts for manually fine-tuning BERT-NLI models for Text Classification. Ex: classifying tasks of Large Language Models (LLM).

## Why Fine-Tune BERT-NLI Models?

Fine-tuning BERT-NLI models allows for improved task classification accuracy in LLMs by adapting the model to specific tasks and datasets. This process leverages the pre-trained knowledge of BERT-NLI models and refines it for enhanced performance.

## How to Run the Fine-Tuning Script?

- Ensure you have all the required libraries installed (`transformers`, `datasets`, `pandas`, etc.).
- Run the `manual_bert_nli_finetune.py` script with the necessary arguments.
  - Example: `python manual_bert_nli_finetune.py --dataset_path "LLM_NLI_Dataset_Balanced.csv" --model_name "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"`
- The script will process the dataset, train the model, and output the fine-tuning results.

## How to Run Unit Tests?

- The `test_manual_bert_nli_finetune.py` script provides basic unit tests to validate the functionality of the fine-tuning process.
- Run the test script using a Python test runner or directly with `python test_manual_bert_nli_finetune.py --dataset_path "LLM_NLI_Dataset_Balanced.csv" --model_name "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"`.
