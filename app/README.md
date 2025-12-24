---
license: mit
tags:
- futures-prediction
- multi-dimensional
- mixture-of-experts
- state-space-model
---

# Futures Prediction Model (MoE + SSM + FiLM)

This repository contains the code and trained weights for a novel architecture designed for multi-dimensional futures prediction. The model was trained on the `futures_dataset_v2.json` dataset.

## Model Description

The model architecture is a combination of:

*   **Mixture of Experts (MoE):** To handle the multi-dimensional nature of futures scenarios.
*   **State Space Model (SSM):** To capture the temporal evolution of futures.
*   **FiLM Conditioning:** To modulate the model's behavior based on the different future axes.

The model is trained to predict a 12-dimensional vector of weights, each corresponding to a different future "axis".

## How to Use

To use this model, you will need to have PyTorch installed. You can then use the `load_model.py` script to load the model and tokenizer.

```python
from load_model import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer()

text = "In a future dominated by hyper-automation, societal structures adapt to new forms of labor and community."
token_ids = tokenizer.encode(text)
tokens_tensor = torch.LongTensor(token_ids).unsqueeze(0)

with torch.no_grad():
    axis_logits, _, _ = model(tokens_tensor)
    axis_predictions = torch.sigmoid(axis_logits)

print(axis_predictions)
```

## Training Data

The model was trained on the `futures_dataset_v2.json` dataset, which contains 3,000 rich, multi-dimensional futures scenarios.

## Training Procedure

The model was trained for 100 epochs with a batch size of 16 and a learning rate of 1e-4. The training script `train_futures_model.py` is available in the original repository.

## Citing

If you use this model or code, please cite:

```
@article{futures-representation-learning,
  title={Learning Multi-Dimensional Futures Representations with Mixture-of-Experts and State Space Models},
  author={Your Name},
  year={2024}
}
```
