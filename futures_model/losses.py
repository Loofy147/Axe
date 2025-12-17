import torch
import torch.nn.functional as F

def ece_loss(logits, labels, n_bins=15, min_bin_size=10):
    """ECE with minimum bin size threshold"""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels).float()

    ece = torch.tensor(0.0, device=logits.device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        n_in_bin = in_bin.sum().item()

        if n_in_bin >= min_bin_size:  # Only compute if enough samples
            accuracy_in_bin = accuracies[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(confidence_in_bin - accuracy_in_bin) * (n_in_bin / len(labels))

    return ece
