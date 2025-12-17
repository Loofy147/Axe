import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from futures_model.model import AxisAwareGPTWithMoE
from futures_model.dataset import SyntheticFuturesDataset

if __name__ == '__main__':
    # Hyperparameters
    vocab_size, d_model, seq_len = 100, 256, 20
    n_axes, n_paths_per_axis, batch_size, num_epochs = 4, 3, 16, 15

    # Model, Loss, Optimizer
    model = AxisAwareGPTWithMoE(vocab_size, d_model)
    criterion_nll = nn.CrossEntropyLoss()
    criterion_axis = nn.CrossEntropyLoss()
    criterion_traj = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Data
    dataloader = DataLoader(SyntheticFuturesDataset(), batch_size=batch_size, shuffle=True)

    # Training
    final_loss = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for tokens, axis_ids in dataloader:
            optimizer.zero_grad()

            logits, axis_pred, traj_states, gate_loss, _, _, _ = model(tokens, axis_ids)

            loss_nll = criterion_nll(logits.view(-1, vocab_size), tokens.view(-1))
            loss_axis = criterion_axis(axis_pred.mean(dim=1), axis_ids)
            loss_traj = criterion_traj(traj_states[:, :-1, :], traj_states.detach()[:, 1:, :])

            loss = loss_nll + 1.0 * loss_axis + 0.1 * loss_traj + 0.01 * gate_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        final_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {final_loss:.4f}")

    # Verification
    print("\n--- Verification ---")
    loss_threshold = 6.5
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss threshold: {loss_threshold}")
    assert final_loss < loss_threshold, "Training loss did not reach the threshold!"
    print("Training loss is below the threshold. Test passed!")
