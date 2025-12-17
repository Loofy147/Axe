"""
Improved Phase 1-5 Training Script with All Fixes
- Fixed adversarial training
- Fixed axis inference
- Added trajectory prediction loss
- Added gate stability penalty
- Added validation split
- Added learning rate scheduling
- Added gradient clipping
- Added logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import math
import torch.nn.functional as F

# Import improved modules
from futures_model.model_improved import AxisAwareGPTWithMoEImproved
from futures_model.dataset import SyntheticFuturesDataset
from futures_model.adversarial_fixed import adversarial_step
from futures_model.losses import ece_loss


def validate(model, val_loader, vocab_size):
    """Run validation and return metrics"""
    model.eval()
    total_loss = 0
    correct_axis = 0
    total_samples = 0

    criterion_nll = nn.CrossEntropyLoss()

    with torch.no_grad():
        for tokens, axis_ids in val_loader:
            logits, axis_pred, _, _, _, _, _, _ = model(tokens, axis_ids)

            # NLL loss
            loss = criterion_nll(logits.view(-1, vocab_size), tokens.view(-1))
            total_loss += loss.item()

            # Axis accuracy
            predicted_axis = axis_pred.mean(dim=1).argmax(dim=1)
            correct_axis += (predicted_axis == axis_ids).sum().item()
            total_samples += axis_ids.size(0)

    model.train()

    return {
        'val_loss': total_loss / len(val_loader),
        'axis_accuracy': correct_axis / total_samples
    }


def train_comprehensive(
    phase='all',
    vocab_size=100,
    d_model=256,
    n_axes=4,
    n_paths_per_axis=3,
    seq_len=20,
    batch_size=16,
    num_epochs=15,
    learning_rate=1e-4,
    use_adversarial=True,
    save_checkpoint=True
):
    """
    Comprehensive training with all improvements.

    Args:
        phase: Which training phase ('1', '2', '3', '4', '5', or 'all')
        use_adversarial: Whether to include adversarial training
    """

    print(f"=== Training Phase {phase} ===\n")

    # Model
    model = AxisAwareGPTWithMoEImproved(
        vocab_size, d_model, n_axes, n_paths_per_axis
    )

    # Losses
    criterion_nll = nn.CrossEntropyLoss()
    criterion_axis = nn.CrossEntropyLoss()

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler
    from torch.optim.lr_scheduler import OneCycleLR

    # Dataset with train/val split
    full_dataset = SyntheticFuturesDataset(
        num_samples=1000, seq_len=seq_len, vocab_size=vocab_size,
        n_axes=n_axes, n_paths_per_axis=n_paths_per_axis
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Loss weights
    beta_axis = 1.0
    beta_traj = 0.1
    beta_gate_entropy = 0.01
    beta_gate_stability = 0.05
    beta_ece = 0.5
    beta_adversarial = 0.5

    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (tokens, axis_ids) in enumerate(train_loader):
            optimizer.zero_grad()

            # Standard forward pass
            (logits, axis_pred, traj_states, gate_entropy_loss,
             gate_stability_loss, uncertainty, temperature, inferred_axis) = model(tokens, axis_ids)

            # Compute losses
            loss_nll = criterion_nll(logits.view(-1, vocab_size), tokens.view(-1))
            loss_axis = criterion_axis(axis_pred.mean(dim=1), axis_ids)

            # Trajectory prediction loss
            loss_traj = model.compute_trajectory_loss(traj_states)

            # Axis inference loss
            loss_inferred = criterion_axis(inferred_axis.mean(dim=1), axis_ids)

            # Calibration loss (if in phase 4+)
            loss_ece = torch.tensor(0.0)
            if phase in ['4', '5', 'all']:
                calibrated_logits = logits / temperature.unsqueeze(-1)
                loss_ece = ece_loss(
                    calibrated_logits.view(-1, vocab_size),
                    tokens.view(-1)
                )

            # Combine losses
            loss = (
                loss_nll +
                beta_axis * loss_axis +
                beta_traj * loss_traj +
                beta_gate_entropy * gate_entropy_loss +
                beta_gate_stability * gate_stability_loss +
                beta_ece * loss_ece +
                0.5 * loss_inferred
            )

            # Adversarial training (if enabled and phase 5)
            if use_adversarial and phase in ['5', 'all']:
                adv_total, adv_clean, adv_robust = adversarial_step(
                    model, tokens, axis_ids, epsilon=0.1
                )
                loss = loss + beta_adversarial * adv_robust

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Epoch summary
        avg_train_loss = total_loss / len(train_loader)
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validation
        val_metrics = validate(model, val_loader, vocab_size)
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Axis Accuracy: {val_metrics['axis_accuracy']:.2%}\n")

        # Early stopping
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0

            if save_checkpoint:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, f'checkpoint_phase{phase}_best.pt')
                print(f"✓ Saved checkpoint (val_loss improved to {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered (patience={patience})")
                break

    print("\n=== Training Complete ===")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # Final evaluation
    final_metrics = validate(model, val_loader, vocab_size)
    print("\n--- Final Metrics ---")
    print(f"Val Loss: {final_metrics['val_loss']:.4f}")
    print(f"Axis Accuracy: {final_metrics['axis_accuracy']:.2%}")

    return model, final_metrics


if __name__ == '__main__':
    # Train all phases
    model, metrics = train_comprehensive(
        phase='all',
        num_epochs=15,
        use_adversarial=True,
        save_checkpoint=True
    )

    print("\n✓ Training completed successfully!")
    print(f"Final validation loss: {metrics['val_loss']:.4f}")
    print(f"Final axis accuracy: {metrics['axis_accuracy']:.2%}")
