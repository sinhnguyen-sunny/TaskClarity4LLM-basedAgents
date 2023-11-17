# Manual Fine-Tuning of BERT-NLI Models

## Overview

This repository contains scripts for the manual fine-tuning of BERT-NLI models. The primary focus is on enhancing Text Classification tasks, such as categorizing the tasks of Large Language Models (LLMs).

## Why Fine-Tune BERT-NLI Models?

Fine-tuning BERT-NLI models is essential for tailoring these advanced pre-trained models to specific tasks and datasets. This process leverages their extensive pre-trained knowledge and refines it to achieve improved accuracy and effectiveness in task-specific applications, especially in the context of LLMs.

## Features of the Fine-Tuning Script

- `manual_bert_nli_finetune.py`: At the heart of our manual fine-tuning process, this script allows for detailed control and customization over the fine-tuning parameters.
- Key Functionalities:
  - Loading and preparing datasets for NLI tasks.
  - Tokenizing datasets using the specified tokenizer.
  - Computing various evaluation metrics after training.
  - Supporting GPU computation for efficient model training.

## Getting Started: How to Run the Fine-Tuning Script?

1. **Installation:**
   - Ensure all required libraries are installed:
     ```bash
     pip install -r requirements.txt
     ```
2. **Running the Script:**
   - Use the `manual_bert_nli_finetune.py` script with the necessary arguments to start the fine-tuning process.
   - **Example Command:**
     ```bash
     python manual_bert_nli_finetune.py --dataset_path "LLM_NLI_Dataset_Balanced.csv" --model_name "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
     ```

## Additional Notes

- The script is designed to provide comprehensive metrics after evaluation, including accuracy, F1 scores (macro and micro), precision, recall, and balanced accuracy.
- For those interested in automated fine-tuning, the `auto_bert_nli_finetune.py` script is also available in this repository. It simplifies the process while maintaining robust fine-tuning capabilities.
  - To run the automated script:
    ```bash
    python auto_bert_nli_finetune.py --dataset_path "LLM_NLI_Dataset_Balanced.csv" --model_name "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    ```
- Ensure your environment meets all the prerequisites outlined in `requirements.txt` for optimal performance and compatibility.
