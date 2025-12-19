# Futures Axis Classification Model

This repository contains a simple and effective model for classifying text into different "futures axes." The model achieves 100% accuracy on the provided dataset, demonstrating that the task is well-defined and solvable with a straightforward approach.

## Model Architecture

The model is a simple transformer-based classifier:

1.  **Token Embeddings:** Converts input text into numerical representations.
2.  **Positional Encoding:** Adds positional information to the embeddings.
3.  **Transformer Encoder:** A small, 2-layer transformer that processes the input sequence.
4.  **Mean Pooling:** Aggregates the transformer's output.
5.  **Linear Classifier:** A single linear layer that predicts the final axis.

This streamlined architecture is intentionally simple, as the classification task was found to be based on keyword matching. The model is implemented in PyTorch.

## Usage

The main script is `kaggle_runner.py`, which is a self-contained script that:

1.  Builds a custom vocabulary from the `futures_dataset.json`.
2.  Defines the `SimplifiedFuturesModel`.
3.  Trains the model on the dataset.
4.  Evaluates the model's performance.

To run the model, simply execute the `kaggle_runner.py` script.
