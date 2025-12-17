import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from futures_model.model import AxisAwareGPTWithMoE
from futures_model.dataset import SyntheticFuturesDataset
from futures_model.adversarial import adversarial_step

if __name__ == '__main__':
    vocab_size, batch_size, num_epochs = 100, 16, 3
    model = AxisAwareGPTWithMoE(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DataLoader(SyntheticFuturesDataset(vocab_size=vocab_size), batch_size=batch_size, shuffle=True)

    final_loss = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for tokens, axis_ids in dataloader:
            optimizer.zero_grad()

            loss_clean, loss_robust = adversarial_step(model, tokens, axis_ids)
            loss = loss_clean + 0.5 * loss_robust
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        final_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {final_loss:.4f}")

    # Verification
    print("\n--- Verification ---")
    loss_threshold = 5.0
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss threshold: {loss_threshold}")
    assert final_loss < loss_threshold, "Training loss did not reach the threshold!"
    print("Training loss is below the threshold. Test passed!")
