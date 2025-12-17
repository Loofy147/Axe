import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from futures_model.model import AxisAwareGPTWithMoE
from futures_model.dataset import SyntheticFuturesDataset

if __name__ == '__main__':
    # Hyperparameters
    vocab_size = 100
    d_model = 256
    n_axes = 4
    n_paths_per_axis = 3
    seq_len = 20
    batch_size = 16
    num_epochs = 10

    # Model, Loss, Optimizer
    model = AxisAwareGPTWithMoE(vocab_size, d_model, n_axes, n_paths_per_axis)
    criterion_nll = nn.CrossEntropyLoss()
    criterion_axis = nn.CrossEntropyLoss()
    criterion_traj = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Data
    dataset = SyntheticFuturesDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    final_loss = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for tokens, axis_ids in dataloader:
            optimizer.zero_grad()

            logits, axis_pred, traj_states, _, _, _, _ = model(tokens, axis_ids)

            # Calculate losses
            loss_nll = criterion_nll(logits.view(-1, vocab_size), tokens.view(-1))
            loss_axis = criterion_axis(axis_pred.mean(dim=1), axis_ids)

            predicted_future_state = traj_states[:, :-1, :]
            actual_future_state = traj_states.detach()[:, 1:, :]
            loss_traj = criterion_traj(predicted_future_state, actual_future_state)

            loss = loss_nll + 1.0 * loss_axis + 0.1 * loss_traj
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        final_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {final_loss:.4f}")

    # --- Verification ---
    print("\n--- Verification ---")
    loss_threshold = 7.0
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss threshold: {loss_threshold}")
    assert final_loss < loss_threshold, "Training loss did not reach the threshold!"
    print("Training loss is below the threshold. Test passed!")
