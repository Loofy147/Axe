import torch.nn.functional as F

def adversarial_step(model, tokens, axis, epsilon=0.1, T=1.0):
    # Get embeddings and enable gradient retention
    embeddings = model.get_embeddings(tokens)
    embeddings.retain_grad()

    # Forward pass with clean embeddings
    logits_clean, _, _, _, _, _, _ = model.forward_from_embeddings(embeddings, axis)
    loss_clean = F.cross_entropy(logits_clean.view(-1, logits_clean.size(-1)), tokens.view(-1))

    # Calculate gradients
    loss_clean.backward(retain_graph=True)

    # PGD attack
    delta = epsilon * embeddings.grad.sign()
    embeddings_adv = embeddings + delta

    # Forward pass with adversarial embeddings
    logits_adv, _, _, _, _, _, _ = model.forward_from_embeddings(embeddings_adv, axis)

    # KL divergence between clean and adversarial
    loss_robust = F.kl_div(
        F.log_softmax(logits_adv / T, dim=-1),
        F.softmax(logits_clean / T, dim=-1),
        reduction='batchmean'
    )

    return loss_clean, loss_robust
