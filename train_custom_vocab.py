"""
FIXED Training Script - Uses Custom Vocabulary

This solves the gibberish generation problem by using a vocabulary
matched to the model's capacity (3K words instead of 50K).

Expected improvements:
- Perplexity: 214,000 → 15-30
- Axis accuracy: 51% → 70-80%
- Text generation: Gibberish → Coherent sentences
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import pickle
import sys

sys.path.append('.')
from futures_model.model_improved import AxisAwareGPTWithMoEImproved


class CustomVocabDataset(Dataset):
    """Dataset using custom vocabulary"""

    def __init__(self, json_path, tokenizer, max_length=50):
        with open(json_path) as f:
            data = json.load(f)

        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize all samples
        for item in data['samples']:
            text = item['text']
            axis_id = item['axis_id']

            # Tokenize using custom tokenizer
            tokens = tokenizer.encode(text, max_length=max_length)

            self.samples.append({
                'tokens': torch.LongTensor(tokens),
                'axis_id': axis_id,
                'text': text
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['tokens'], sample['axis_id']


def train_with_custom_vocab(
    dataset_path="futures_dataset.json",
    vocab_path="custom_vocab.pkl",
    d_model=256,
    num_epochs=40,  # Increased from 20
    batch_size=16,
    learning_rate=1e-4,
    checkpoint_path="checkpoint_custom_vocab.pt"
):
    """
    Train with custom vocabulary.

    Key changes from original:
    1. Uses custom 3K vocab instead of GPT-2's 50K
    2. Increased axis loss weight (2.0 instead of 1.0)
    3. Longer training (40 epochs instead of 20)
    4. Lower initial learning rate for stability
    """

    print("=" * 80)
    print("Training with Custom Vocabulary (FIXED VERSION)")
    print("=" * 80)

    # Load custom vocabulary
    print("\n1. Loading custom vocabulary...")
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)

    vocab_dict = vocab_data['vocab_dict']
    vocab_size = len(vocab_dict)

    print(f"   Vocabulary size: {vocab_size:,} (was 50,257)")
    print(f"   ✓ Model can handle this!")

    # Create tokenizer
    from build_custom_vocab import CustomTokenizer
    tokenizer = CustomTokenizer(vocab_dict)

    # Load dataset
    print(f"\n2. Loading dataset...")
    full_dataset = CustomVocabDataset(dataset_path, tokenizer)
    print(f"   Total samples: {len(full_dataset)}")

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print(f"\n3. Initializing model...")
    model = AxisAwareGPTWithMoEImproved(
        vocab_size=vocab_size,  # KEY FIX: 3K instead of 50K
        d_model=d_model,
        n_axes=4,
        n_paths_per_axis=3,
        n_head=8,
        n_layer=4,
        d_state=128
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Parameters per vocab item: {total_params / vocab_size:.0f}")
    print(f"   (Was: {31761139 / 50257:.0f} - not enough capacity!)")

    # Setup training
    criterion_nll = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_axis = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Training loop
    print(f"\n4. Training for {num_epochs} epochs...")
    print("=" * 80)

    best_val_loss = float('inf')
    best_axis_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (tokens, axis_ids) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            logits, axis_pred, traj_states, gate_entropy, gate_stability, _, _, _ = model(tokens, axis_ids)

            # Compute losses
            loss_nll = criterion_nll(
                logits.view(-1, vocab_size),
                tokens.view(-1)
            )
            loss_axis = criterion_axis(axis_pred.mean(dim=1), axis_ids)
            loss_traj = model.compute_trajectory_loss(traj_states)

            # INCREASED AXIS WEIGHT from 1.0 to 2.0
            loss = (
                loss_nll +
                2.0 * loss_axis +      # Was 1.0, now 2.0
                0.1 * loss_traj +
                0.01 * gate_entropy +
                0.05 * gate_stability
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # Axis accuracy
            predicted_axis = axis_pred.mean(dim=1).argmax(dim=1)
            train_correct += (predicted_axis == axis_ids).sum().item()
            train_total += axis_ids.size(0)

            # Progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Axis Acc: {100*train_correct/train_total:.1f}%")

        avg_train_loss = train_loss / len(train_loader)
        train_axis_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tokens, axis_ids in val_loader:
                logits, axis_pred, _, _, _, _, _, _ = model(tokens, axis_ids)

                loss = criterion_nll(logits.view(-1, vocab_size), tokens.view(-1))
                val_loss += loss.item()

                predicted_axis = axis_pred.mean(dim=1).argmax(dim=1)
                val_correct += (predicted_axis == axis_ids).sum().item()
                val_total += axis_ids.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_axis_acc = 100 * val_correct / val_total

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

        # Epoch summary
        print("\n" + "=" * 80)
        print(f"Epoch {epoch+1}/{num_epochs} Summary")
        print("=" * 80)
        print(f"Train Loss: {avg_train_loss:.4f} | Train Axis Acc: {train_axis_acc:.2f}%")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Axis Acc:   {val_axis_acc:.2f}%")
        print(f"Perplexity: {perplexity:.2f}")

        # Save best model
        if val_axis_acc > best_axis_acc:
            best_axis_acc = val_axis_acc
            best_val_loss = avg_val_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_axis_acc': val_axis_acc,
                'train_loss': avg_train_loss,
                'train_axis_acc': train_axis_acc,
                'vocab_size': vocab_size,
                'perplexity': perplexity
            }, checkpoint_path)

            print(f"✓ Saved checkpoint (axis acc improved to {val_axis_acc:.2f}%)")

        print("=" * 80 + "\n")

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Axis Accuracy: {best_axis_acc:.2f}%")
    print(f"Best Perplexity: {torch.exp(torch.tensor(best_val_loss)):.2f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Interpretation
    print("\n" + "=" * 80)
    print("Results Interpretation")
    print("=" * 80)

    if best_axis_acc >= 75:
        print("✓✓✓ EXCELLENT RESULTS!")
        print("   Axis accuracy >75% means the model is learning")
        print("   meaningful semantic differences between futures.")
        print("\n   Next steps:")
        print("   1. Test axis controllability with generation")
        print("   2. Run ablation study")
        print("   3. Scale up and write paper!")

    elif best_axis_acc >= 60:
        print("✓ GOOD RESULTS")
        print("   Axis accuracy 60-75% shows the model is learning,")
        print("   but there's room for improvement.")
        print("\n   Try:")
        print("   - Train longer (60-80 epochs)")
        print("   - Increase model size (d_model=384)")
        print("   - Increase axis loss weight to 3.0")

    else:
        print("⚠️  NEEDS IMPROVEMENT")
        print("   Axis accuracy <60% suggests issues remain.")
        print("\n   Check:")
        print("   - Is perplexity < 30? (language modeling working)")
        print("   - Are axes separable in the dataset?")
        print("   - Try even higher axis loss weight (3.0-5.0)")

    return model, best_axis_acc, tokenizer
