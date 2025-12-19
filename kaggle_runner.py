"""
This script consolidates the training and testing pipelines for the futures model,
designed to be run in a Kaggle environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
from pathlib import Path
import sys
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from collections import Counter
import pickle

def build_custom_vocabulary(dataset_path='futures_dataset.json', vocab_size=3000):
    """
    Build custom vocabulary from dataset.

    Args:
        dataset_path: Path to futures dataset JSON
        vocab_size: Target vocabulary size (not including special tokens)

    Returns:
        vocab_dict: {word: index} mapping
        idx_to_word: {index: word} mapping
    """

    print("=" * 70)
    print("Building Custom Vocabulary")
    print("=" * 70)

    # Load dataset
    print(f"\n1. Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        data = json.load(f)

    samples = data['samples']
    print(f"   Total samples: {len(samples)}")

    # Tokenize and count words
    print("\n2. Counting words...")
    word_counts = Counter()

    for sample in samples:
        text = sample['text'].lower()

        # Simple tokenization (split on whitespace and punctuation)
        text = text.replace(',', ' ,').replace('.', ' .')
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        text = text.replace('"', ' " ').replace("'", " ' ")

        words = text.split()
        word_counts.update(words)

    total_words = sum(word_counts.values())
    unique_words = len(word_counts)
    print(f"   Total words: {total_words:,}")
    print(f"   Unique words: {unique_words:,}")

    # Create vocabulary
    print(f"\n3. Creating vocabulary (target size: {vocab_size})...")

    # Special tokens
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']

    # Top N most common words
    top_words = [word for word, _ in word_counts.most_common(vocab_size)]

    # Combine
    vocabulary = special_tokens + top_words
    actual_vocab_size = len(vocabulary)

    # Calculate coverage
    top_word_count = sum(word_counts[w] for w in top_words)
    coverage = 100 * top_word_count / total_words

    print(f"   Final vocabulary size: {actual_vocab_size:,}")
    print(f"   Coverage: {coverage:.2f}% of dataset tokens")

    # Create mappings
    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}
    idx_to_word = {idx: word for word, idx in vocab_dict.items()}

    # Save vocabulary
    print("\n4. Saving vocabulary...")

    with open('custom_vocab.json', 'w') as f:
        json.dump(vocab_dict, f, indent=2)

    with open('custom_vocab.pkl', 'wb') as f:
        pickle.dump({
            'vocab_dict': vocab_dict,
            'idx_to_word': idx_to_word,
            'word_counts': dict(word_counts.most_common(vocab_size)),
            'special_tokens': special_tokens
        }, f)

    print(f"   Saved to: custom_vocab.json and custom_vocab.pkl")

    # Statistics
    print("\n" + "=" * 70)
    print("Vocabulary Statistics")
    print("=" * 70)
    print(f"Vocabulary size: {actual_vocab_size:,}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Unknown token rate: {100 - coverage:.2f}%")

    # Show top 20 words
    print("\nTop 20 most common words:")
    for i, (word, count) in enumerate(word_counts.most_common(20), 1):
        print(f"  {i:2d}. {word:20s} ({count:4d} occurrences)")

    # Show some axis-specific words
    print("\nSample vocabulary by category:")
    tech_words = [w for w in top_words if any(kw in w for kw in ['ai', 'robot', 'automat', 'algorithm', 'machine', 'digital'])][:10]
    env_words = [w for w in top_words if any(kw in w for kw in ['climate', 'environment', 'carbon', 'energy', 'sustain', 'ecosystem'])][:10]
    society_words = [w for w in top_words if any(kw in w for kw in ['community', 'individual', 'global', 'collective', 'society'])][:10]

    if tech_words:
        print(f"\n  Tech words: {', '.join(tech_words)}")
    if env_words:
        print(f"  Environment words: {', '.join(env_words)}")
    if society_words:
        print(f"  Society words: {', '.join(society_words)}")

    print("\n" + "=" * 70)
    print("✓ Vocabulary creation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Use custom_vocab.pkl in training")
    print("  2. Retrain model with vocab_size = " + str(actual_vocab_size))
    print("  3. Expected: Coherent text generation + 70-80% axis accuracy")

    return vocab_dict, idx_to_word


class CustomTokenizer:
    """Simple tokenizer using custom vocabulary"""

    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.idx_to_word = {idx: word for word, idx in vocab_dict.items()}
        self.pad_token_id = vocab_dict['<PAD>']
        self.unk_token_id = vocab_dict['<UNK>']
        self.vocab_size = len(vocab_dict)

    def encode(self, text, max_length=50):
        """Encode text to token IDs"""
        # Simple tokenization
        text = text.lower()
        text = text.replace(',', ' ,').replace('.', ' .')
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        text = text.replace('"', ' " ').replace("'", " ' ")

        words = text.split()

        # Convert to IDs
        token_ids = []
        for word in words:
            if word in self.vocab_dict:
                token_ids.append(self.vocab_dict[word])
            else:
                token_ids.append(self.unk_token_id)

        # Pad or truncate
        if len(token_ids) < max_length:
            token_ids += [self.pad_token_id] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        words = []
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']

        for idx in token_ids:
            if isinstance(idx, int):
                word = self.idx_to_word.get(idx, '<UNK>')
            else:
                word = self.idx_to_word.get(idx.item(), '<UNK>')

            if skip_special_tokens and word in special_tokens:
                continue

            words.append(word)

        # Join with spaces, but handle punctuation
        text = ' '.join(words)
        text = text.replace(' ,', ',').replace(' .', '.')
        text = text.replace('( ', '(').replace(' )', ')')

        return text

# --- Start Embedded Model Definition ---

class AxisMoE(nn.Module):
    def __init__(self, d_model=256, n_experts=8, d_axis_emb=128):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model + d_axis_emb, n_experts)
        self.prev_gates = None  # Track previous gates

    def forward(self, h, a):
        gate_input = torch.cat([h, a.unsqueeze(1).repeat(1, h.size(1), 1)], dim=-1)
        g = F.softmax(self.gate(gate_input), dim=-1)

        # Sparse routing: top-k experts (k=2)
        topk_gates, topk_indices = torch.topk(g, k=2, dim=-1)
        topk_gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True)

        # Simplified and corrected vectorized implementation
        batch_size, seq_len, d_model = h.shape
        h_flat = h.view(-1, d_model) # (batch_size * seq_len, d_model)

        final_output = torch.zeros_like(h_flat)

        for i, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert (top-2)
            mask = (topk_indices == i).any(dim=-1)
            mask_flat = mask.view(-1)

            if mask_flat.any():
                # Get the gate values for the tokens routed to this expert
                gate_values = torch.where(topk_indices == i, topk_gates, torch.zeros_like(topk_gates)).sum(dim=-1)
                gate_values_flat = gate_values.view(-1, 1)

                # Apply the expert to the selected tokens and weight by the gate values
                final_output[mask_flat] += (expert(h_flat[mask_flat]) * gate_values_flat[mask_flat])

        output = final_output.view(batch_size, seq_len, d_model)

        # Compute losses
        entropy_loss = -(g * torch.log(g + 1e-10)).sum(-1).mean()

        stability_loss = 0.0
        if self.training and self.prev_gates is not None:
            stability_loss = F.mse_loss(g, self.prev_gates.detach())

        self.prev_gates = g.detach() if self.training else None

        return output, entropy_loss, stability_loss

class TrajectorySSM(nn.Module):
    def __init__(self, d_model, d_state=128, d_axis_emb=128):
        super().__init__()
        self.d_state = d_state
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Linear(d_axis_emb, d_state)
        self.C = nn.Linear(d_state, d_model)
        self.h_proj = nn.Linear(d_model, d_state)

    def forward(self, h, a, s_t):
        h_projected = self.h_proj(h)
        return torch.tanh(F.linear(s_t, self.A) + self.B(a) * h_projected)

class AxisAwareGPTWithMoEImproved(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_axes=4, n_paths_per_axis=3, n_head=8, n_layer=4, d_state=128, n_experts=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_axis_emb = d_model // 2

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.axis_emb = nn.Embedding(n_axes * n_paths_per_axis, self.d_axis_emb)
        self.film_gamma = nn.Linear(self.d_axis_emb, d_model)
        self.film_beta = nn.Linear(self.d_axis_emb, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layer)

        self.moe_layer = AxisMoE(d_model, n_experts, d_axis_emb=self.d_axis_emb)
        self.trajectory_ssm = TrajectorySSM(d_model, d_state, self.d_axis_emb)
        self.trajectory_predictor = nn.Linear(d_state, d_state)

        # Heads
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.axis_head = nn.Linear(d_model, n_axes * n_paths_per_axis)
        self.axis_inference_head = nn.Linear(d_model, n_axes * n_paths_per_axis)
        self.uncertainty_head = nn.Linear(d_model, 1)
        self.temp_head = nn.Linear(d_model, 1)

    def get_embeddings(self, tokens):
        return self.token_emb(tokens)

    def compute_trajectory_loss(self, trajectory_states):
        predicted_future = self.trajectory_predictor(trajectory_states[:, :-1, :])
        actual_future = trajectory_states[:, 1:, :].detach()
        return F.mse_loss(predicted_future, actual_future)

    def forward_from_embeddings(self, embeddings, axis_id):
        a = self.axis_emb(axis_id)
        gamma = self.film_gamma(a).unsqueeze(1)
        beta = self.film_beta(a).unsqueeze(1)
        x = gamma * embeddings + beta

        h = self.transformer_encoder(x)
        moe_h, gate_loss, stability_loss = self.moe_layer(h, a)
        h = h + moe_h

        s_t = torch.zeros(embeddings.size(0), self.d_state).to(embeddings.device)
        trajectory_states, ssm_outputs = [], []
        for t in range(embeddings.size(1)):
            s_t = self.trajectory_ssm(h[:, t, :], a, s_t)
            trajectory_states.append(s_t)
            ssm_outputs.append(self.trajectory_ssm.C(s_t))

        trajectory_states = torch.stack(trajectory_states, dim=1)

        combined_h = h + torch.stack(ssm_outputs, dim=1)

        logits = self.lm_head(combined_h)
        axis_pred = self.axis_head(combined_h)
        inferred_axis = self.axis_inference_head(combined_h)
        uncertainty = self.uncertainty_head(combined_h).squeeze(-1)
        temperature = torch.sigmoid(self.temp_head(combined_h)).squeeze(-1) * 2 + 0.5

        return logits, axis_pred, trajectory_states, gate_loss, stability_loss, uncertainty, temperature, inferred_axis

    def forward(self, tokens, axis_id):
        embeddings = self.get_embeddings(tokens)
        return self.forward_from_embeddings(embeddings, axis_id)

    def infer_axis(self, tokens):
        """Infer axis from tokens by marginalizing over all possible axes"""
        with torch.no_grad():
            n_axes = self.axis_emb.num_embeddings
            all_inferences = []

            for axis_id in range(n_axes):
                axis_tensor = torch.tensor([axis_id]).to(tokens.device)
                _, _, _, _, _, _, _, inferred = self.forward(
                    tokens.unsqueeze(0) if tokens.dim() == 1 else tokens,
                    axis_tensor
                )
                all_inferences.append(inferred)

            avg_inference = torch.stack(all_inferences).mean(dim=0)
            return torch.argmax(avg_inference.mean(dim=1), dim=1)

# --- End Embedded Model Definition ---

AXIS_NAMES = {
    0: "Tech: Hyper-automation",
    1: "Tech: Human-centric",
    2: "Tech: Resource-abundant",
    3: "Society: Individualistic",
    4: "Society: Community",
    5: "Society: Global",
    6: "Environment: Crisis",
    7: "Environment: Restoration",
    8: "Environment: Adaptation",
    9: "Creativity: Immersive/Digital",
    10: "Creativity: Physical/Tangible",
    11: "Creativity: Collaborative",
}

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

def generate_with_custom_vocab(model, tokenizer, prompt_text, axis_id, max_length=20, temperature=0.8):
    """Generate text with custom vocabulary"""
    model.eval()

    # Tokenize prompt
    prompt_tokens = torch.LongTensor(tokenizer.encode(prompt_text, max_length=50))

    # Find where padding starts
    pad_id = tokenizer.pad_token_id
    actual_length = (prompt_tokens != pad_id).sum().item()
    tokens = prompt_tokens[:actual_length]

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (pad to 50 if needed)
            input_tokens = tokens.clone()
            if len(input_tokens) < 50:
                padding = torch.full((50 - len(input_tokens),), pad_id, dtype=torch.long)
                input_tokens = torch.cat([input_tokens, padding])
            else:
                input_tokens = input_tokens[:50]

            # Forward pass
            logits, _, _, _, _, temp_head, _, _ = model(
                input_tokens.unsqueeze(0),
                torch.tensor([axis_id])
            )

            # Get logits for last real token
            last_real_pos = min(len(tokens) - 1, 49)
            next_token_logits = logits[0, last_real_pos, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, 1)

            # Stop if we generate padding or end token
            if next_token.item() == pad_id or next_token.item() == tokenizer.vocab_dict.get('<END>', -1):
                break

            generated_tokens.append(next_token.item())
            tokens = torch.cat([tokens, next_token])

    # Decode
    full_tokens = prompt_tokens[:actual_length].tolist() + generated_tokens
    generated_text = tokenizer.decode(full_tokens, skip_special_tokens=True)

    return generated_text



if __name__ == "__main__":
    # Define Kaggle paths
    DATASET_PATH = "/kaggle/input/futures-dataset/futures_dataset.json"
    VOCAB_PATH = "/kaggle/working/custom_vocab.pkl"
    CHECKPOINT_PATH = "/kaggle/working/checkpoint.pt"

    # Build vocabulary
    build_custom_vocabulary(dataset_path=DATASET_PATH)

    # Train the model
    model, accuracy, tokenizer = train_with_custom_vocab(
        dataset_path=DATASET_PATH,
        vocab_path=VOCAB_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        d_model=256,
        num_epochs=40
    )

    # Test the model
    if model:
        test_custom_vocab_controllability(
            checkpoint_path=CHECKPOINT_PATH,
            vocab_path=VOCAB_PATH,
            d_model=256
        )
