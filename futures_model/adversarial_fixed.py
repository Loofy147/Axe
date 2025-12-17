import torch
import torch.nn.functional as F

def adversarial_step(model, tokens, axis, epsilon=0.1, T=1.0, beta=0.5):
    model.zero_grad()

    # Forward pass 1: Clean
    embeddings = model.get_embeddings(tokens)

    logits_clean, _, _, _, _, _, _, _ = model.forward_from_embeddings(embeddings, axis)
    loss_clean = F.cross_entropy(logits_clean.view(-1, logits_clean.size(-1)), tokens.view(-1))

    # Compute adversarial perturbation
    # Note: Pytorch will not backprop through get_embeddings, so we need to enable it.
    embeddings.requires_grad_(True)
    grad = torch.autograd.grad(loss_clean, embeddings, create_graph=False)[0]
    delta = epsilon * grad.sign()

    # Forward pass 2: Adversarial (detached)
    embeddings_adv = (embeddings + delta).detach()

    logits_adv, _, _, _, _, _, _, _ = model.forward_from_embeddings(embeddings_adv, axis)

    # KL divergence for robustness
    loss_robust = F.kl_div(
        F.log_softmax(logits_adv / T, dim=-1),
        F.softmax(logits_clean.detach() / T, dim=-1),
        reduction='batchmean'
    )

    # Combined loss
    total_loss = loss_clean + beta * loss_robust
    return total_loss, loss_clean.item(), loss_robust.item()
