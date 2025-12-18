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

class RealFuturesDataset(Dataset):
    """Dataset loading from JSON with GPT-2 tokenization."""

    def __init__(self, json_path, tokenizer, max_length=50):
        with open(json_path) as f:
            data = json.load(f)

        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in data['samples']:
            text = item['text']
            axis_id = item['axis_id']

            tokens = tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )

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

def train_on_real_data(
    dataset_path="futures_dataset.json",
    vocab_size=50257,
    d_model=512,
    num_epochs=30,
    batch_size=16,
    learning_rate=1e-4,
    checkpoint_path="checkpoint_real_data.pt"
):
    print("=" * 70)
    print("Training on Real Futures Dataset")
    print("=" * 70)

    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return None, None

    print("\n1. Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size: {len(tokenizer):,}")

    print(f"\n2. Loading dataset from {dataset_path}...")
    full_dataset = RealFuturesDataset(dataset_path, tokenizer)
    print(f"   Total samples: {len(full_dataset)}")

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

    print(f"\n3. Initializing model (d_model={d_model})...")
    model = AxisAwareGPTWithMoEImproved(
        vocab_size=vocab_size,
        d_model=d_model,
        n_axes=4,
        n_paths_per_axis=3,
        n_head=8,
        n_layer=4
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

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

    print(f"\n4. Training for {num_epochs} epochs...")
    print("=" * 70)

    best_val_loss = float('inf')
    best_axis_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (tokens, axis_ids) in enumerate(train_loader):
            optimizer.zero_grad()

            logits, axis_pred, traj_states, gate_entropy, gate_stability, _, _, _ = model(tokens, axis_ids)

            loss_nll = criterion_nll(
                logits.view(-1, vocab_size),
                tokens.view(-1)
            )
            loss_axis = criterion_axis(axis_pred.mean(dim=1), axis_ids)
            loss_traj = model.compute_trajectory_loss(traj_states)

            loss = (
                loss_nll +
                2.0 * loss_axis +
                0.1 * loss_traj +
                0.01 * gate_entropy +
                0.05 * gate_stability
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            predicted_axis = axis_pred.mean(dim=1).argmax(dim=1)
            train_correct += (predicted_axis == axis_ids).sum().item()
            train_total += axis_ids.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Axis Acc: {100*train_correct/train_total:.1f}%")

        avg_train_loss = train_loss / len(train_loader)
        train_axis_acc = 100 * train_correct / train_total

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

        print("\n" + "=" * 70)
        print(f"Epoch {epoch+1}/{num_epochs} Summary")
        print("=" * 70)
        print(f"Train Loss: {avg_train_loss:.4f} | Train Axis Acc: {train_axis_acc:.2f}%")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Axis Acc:   {val_axis_acc:.2f}%")

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
            }, checkpoint_path)

            print(f"✓ Saved checkpoint (axis acc improved to {val_axis_acc:.2f}%)")

        print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Axis Accuracy: {best_axis_acc:.2f}%")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return model, best_axis_acc

def generate_with_axis(model, tokenizer, prompt_text, axis_id, max_length=30, temperature=0.8):
    model.eval()

    prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt')[0]
    tokens = prompt_tokens

    with torch.no_grad():
        for _ in range(max_length):
            logits, _, _, _, _, temp_head, _, _ = model(
                tokens.unsqueeze(0),
                torch.tensor([axis_id])
            )

            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            tokens = torch.cat([tokens, next_token])

    generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return generated_text

def test_axis_controllability(
    checkpoint_path="checkpoint_real_data.pt",
    vocab_size=50257,
    d_model=512
):
    print("=" * 80)
    print("Axis Controllability Test")
    print("=" * 80)

    print("\n1. Loading model...")
    model = AxisAwareGPTWithMoEImproved(
        vocab_size=vocab_size,
        d_model=d_model,
        n_axes=4,
        n_paths_per_axis=3
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"   Loaded from: {checkpoint_path}")
    print(f"   Val accuracy: {checkpoint['val_axis_acc']:.2f}%")

    print("\n2. Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_prompts = [
        "In the future, people will",
        "The next generation of technology",
        "Communities are",
        "The environment is becoming",
        "Artists create by",
    ]

    print("\n3. Testing axis control...")
    print("=" * 80)

    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 80)

        generations = {}
        for axis_id in range(12):
            generated = generate_with_axis(
                model, tokenizer, prompt, axis_id,
                max_length=20, temperature=0.8
            )

            continuation = generated[len(prompt):].strip()
            generations[axis_id] = continuation

            axis_name = AXIS_NAMES[axis_id]
            print(f"[{axis_id:2d}] {axis_name:35s}: {continuation[:60]}")

        unique_continuations = len(set(generations.values()))
        print(f"\nUnique continuations: {unique_continuations}/12")

        if unique_continuations < 4:
            print("⚠️  WARNING: Very low diversity - axis control may not be working")
        elif unique_continuations < 8:
            print("⚠️  Moderate diversity - some axis collapse may be occurring")
        else:
            print("✓ Good diversity - axes appear to be controlling generation")

if __name__ == "__main__":
    # Define Kaggle paths
    DATASET_PATH = "/kaggle/input/futures-dataset/futures_dataset.json"
    CHECKPOINT_PATH = "/kaggle/working/checkpoint.pt"

    # Train the model
    model, accuracy = train_on_real_data(
        dataset_path=DATASET_PATH,
        checkpoint_path=CHECKPOINT_PATH
    )

    # Test the model
    if model:
        test_axis_controllability(
            checkpoint_path=CHECKPOINT_PATH
        )
