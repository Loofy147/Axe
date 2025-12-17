import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from futures_model.model import AxisAwareGPTWithMoE
from futures_model.dataset import SyntheticFuturesDataset
import math
import torch.nn.functional as F

def axis_aware_beam_search(model, prompt_tokens, target_axis, k=5, lambda_axis=0.1, max_len=20):
    model.eval()
    beams = [(prompt_tokens, 0.0, 0.0)]  # (tokens, score, total_uncertainty)

    with torch.no_grad():
        for _ in range(max_len):
            candidates = []
            for tokens, score, total_uncertainty in beams:
                logits, _, _, _, uncertainty, temp, _ = model(tokens.unsqueeze(0), target_axis)

                last_logits = logits[0, -1, :]
                last_temp = temp[0, -1]
                last_uncertainty = uncertainty[0, -1]

                probs = F.softmax(last_logits / last_temp, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, k=10)

                for prob, tok_id in zip(topk_probs, topk_ids):
                    new_tokens = torch.cat([tokens, tok_id.unsqueeze(0)])
                    new_score = score + math.log(prob)

                    inferred_axis = model.infer_axis(new_tokens)
                    axis_div = F.mse_loss(inferred_axis.float(), target_axis.float())
                    new_score -= lambda_axis * axis_div

                    new_uncertainty = total_uncertainty + last_uncertainty.item()
                    candidates.append((new_tokens, new_score, new_uncertainty))

            reranked_candidates = sorted(candidates, key=lambda x: x[1] / (1 + x[2] / len(x[0])), reverse=True)
            beams = reranked_candidates[:k]

    return beams[0][0]

def ece_loss(logits, labels, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

if __name__ == '__main__':
    # Hyperparameters
    vocab_size, d_model, seq_len = 100, 256, 20
    n_axes, n_paths_per_axis, batch_size, num_epochs = 4, 3, 16, 10

    # Model, Loss, Optimizer
    model = AxisAwareGPTWithMoE(vocab_size, d_model)
    criterion_nll = nn.CrossEntropyLoss()
    criterion_axis = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Data
    dataloader = DataLoader(SyntheticFuturesDataset(), batch_size=batch_size, shuffle=True)

    # Training
    for epoch in range(num_epochs):
        total_loss = 0
        for tokens, axis_ids in dataloader:
            optimizer.zero_grad()

            logits, axis_pred, _, _, _, temp, inferred_axis = model(tokens, axis_ids)

            loss_nll = criterion_nll(logits.view(-1, vocab_size), tokens.view(-1))
            loss_axis = criterion_axis(axis_pred.mean(dim=1), axis_ids)
            loss_inferred_axis = criterion_axis(inferred_axis.mean(dim=1), axis_ids)

            calibrated_logits = logits / temp.unsqueeze(-1)
            loss_ece = ece_loss(calibrated_logits.view(-1, vocab_size), tokens.view(-1))

            loss = loss_nll + 1.0 * loss_axis + 0.5 * loss_inferred_axis + 1.0 * loss_ece
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Verification
    print("\n--- Verification ---")
    prompt = torch.LongTensor([10, 20, 30])
    target_axis = torch.LongTensor([5])

    generated_tokens = axis_aware_beam_search(model, prompt, target_axis)

    print(f"Prompt: {prompt.tolist()}")
    print(f"Target Axis: {target_axis.item()}")
    print(f"Generated Sequence: {generated_tokens.tolist()}")

    assert len(generated_tokens) > len(prompt), "Decoding failed!"
    print("Training and decoding with uncertainty heads successful!")
