# Robust Futures Axis Classification Model

This repository contains a robust and reproducible pipeline for training a futures-axis classification model that aims for conceptual understanding rather than simple keyword matching.

## The Challenge: Beyond Keyword Matching

Initial experiments revealed that a simple model could achieve 100% accuracy on the original dataset by memorizing keywords. This indicated that the model was not learning the underlying concepts of the "futures axes." The goal of this project was to address this by developing a more robust model and a more challenging dataset.

## The Solution: A Self-Contained, Robust Pipeline

The `kaggle_runner.py` script is a self-contained pipeline that implements a three-pronged approach to this challenge:

1.  **Automated Data Augmentation:** The script first checks for the existence of an `augmented_futures_dataset.json`. If it's not found, it automatically generates one by replacing keywords in the original `futures_dataset.json` with synonyms. This creates a more diverse and challenging dataset that discourages keyword memorization.

2.  **Self-Supervised Learning:** The model is trained on a combined objective of axis classification and Masked Language Modeling (MLM). The MLM task forces the model to learn the contextual relationships between words, which encourages a deeper semantic understanding of the text.

3.  **Keyword Robustness Analysis:** After training, the script runs a keyword robustness analysis. This analysis measures how much the model's predictions change when keywords are removed from the text, providing a quantitative measure of its conceptual understanding.

## Model Architecture

The model is a `SimplifiedFuturesModel` with two heads:

1.  **Classification Head:** A mean-pooling layer followed by a linear classifier that predicts the futures axis.
2.  **MLM Head:** A linear layer that predicts the original tokens for masked positions in the input.

This dual-head architecture allows the model to learn both the classification and the self-supervised tasks simultaneously.

## Usage

To run the entire pipeline, simply install the dependencies and run the main script:

```bash
pip install -r requirements.txt
python3 kaggle_runner.py
```

The script will handle the data augmentation, training, and evaluation automatically.
