"""
SIMPLIFIED MODEL - Remove ALL Complexity

Since TF-IDF gets 100% accuracy, we know:
1. The task is trivial (just pattern matching keywords)
2. The complex architecture (MoE, SSM, FiLM) is completely unnecessary
3. We just need: Embeddings → Transformer → Classification head

This should easily get 70-80%+ accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import json
from collections import Counter
import pickle
import torch.optim as optim

class CustomTokenizer:
    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.idx_to_word = {idx: word for word, idx in vocab_dict.items()}
        self.pad_token_id = vocab_dict['<PAD>']
        self.unk_token_id = vocab_dict['<UNK>']
        self.vocab_size = len(vocab_dict)

    def encode(self, text, max_length=50):
        text = text.lower()
        text = text.replace(',', ' ,').replace('.', ' .')
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        text = text.replace('"', ' " ').replace("'", " ' ")
        words = text.split()

        token_ids = []
        for word in words:
            token_ids.append(self.vocab_dict.get(word, self.unk_token_id))

        if len(token_ids) < max_length:
            token_ids += [self.pad_token_id] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
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

        text = ' '.join(words)
        text = text.replace(' ,', ',').replace(' .', '.')
        text = text.replace('( ', '(').replace(' )', ')')
        return text


class SimplifiedFuturesModel(nn.Module):
    """
    ULTRA-SIMPLE model for axis classification.

    Architecture:
    1. Token embeddings
    2. Positional encoding
    3. Small transformer (2 layers)
    4. Mean pooling
    5. Classification head

    NO MoE, NO SSM, NO FiLM, NO complexity!
    """

    def __init__(self, vocab_size, n_classes=12, d_model=128, n_head=4, n_layers=2):
        super().__init__()

        self.d_model = d_model

        # Simple embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(50, d_model)  # Max sequence length 50

        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head (just one layer!)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, tokens):
        # Embeddings
        batch_size, seq_len = tokens.shape
        token_emb = self.token_emb(tokens)

        # Positional encoding
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)

        x = token_emb + pos_emb

        # Transformer
        x = self.transformer(x)

        # Mean pooling (ignore padding)
        mask = (tokens != 0).float()  # Assuming 0 is padding
        x_masked = x * mask.unsqueeze(-1)
        x_pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Classification
        logits = self.classifier(x_pooled)

        return logits


def build_custom_vocabulary(dataset_path, vocab_size=3000):
    print("Building vocabulary...")
    with open(dataset_path) as f:
        data = json.load(f)

    word_counts = Counter()
    for sample in data['samples']:
        text = sample['text'].lower()
        text = text.replace(',', ' ,').replace('.', ' .')
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        words = text.split()
        word_counts.update(words)

    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    top_words = [word for word, _ in word_counts.most_common(vocab_size)]
    vocabulary = special_tokens + top_words

    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}

    with open('custom_vocab_simple.pkl', 'wb') as f:
        pickle.dump({'vocab_dict': vocab_dict}, f)

    print(f"Vocabulary size: {len(vocab_dict)}")
    return vocab_dict


class CustomVocabDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=50):
        with open(json_path) as f:
            data = json.load(f)

        self.samples = []
        for item in data['samples']:
            tokens = tokenizer.encode(item['text'], max_length=max_length)
            self.samples.append({
                'tokens': torch.LongTensor(tokens),
                'axis_id': item['axis_id']
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['tokens'], sample['axis_id']


def train_simplified_model(
    dataset_path='futures_dataset.json',
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3
):
    """
    Train the SIMPLIFIED model.

    Key changes:
    - Small model (128 dim, 2 layers)
    - ONLY axis classification (no language modeling!)
    - Higher learning rate (convergence faster)
    - Early stopping
    """

    print("=" * 80)
    print("SIMPLIFIED MODEL TRAINING")
    print("=" * 80)
    print()
    print("Changes from previous attempts:")
    print("  - Removed MoE (not needed)")
    print("  - Removed SSM (not needed)")
    print("  - Removed FiLM (not needed)")
    print("  - Removed language modeling (not needed)")
    print("  - Just: Embeddings → Transformer → Classifier")
    print()
    print("Expected: 70-90% accuracy (TF-IDF got 100%!)")
    print("=" * 80)
    print()

    # Build vocab
    vocab_dict = build_custom_vocabulary(dataset_path)
    tokenizer = CustomTokenizer(vocab_dict)
    vocab_size = len(vocab_dict)

    # Load dataset
    full_dataset = CustomVocabDataset(dataset_path, tokenizer)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Initialize model
    model = SimplifiedFuturesModel(
        vocab_size=vocab_size,
        n_classes=12,
        d_model=128,  # Small!
        n_head=4,
        n_layers=2    # Only 2 layers!
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (was 24M!)")
    print()

    # Simple training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    print("Training...")
    print("=" * 80)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for tokens, labels in train_loader:
            optimizer.zero_grad()

            logits = model(tokens)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tokens, labels in val_loader:
                logits = model(tokens)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Acc: {train_acc:5.2f}% | "
              f"Val Acc: {val_acc:5.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'vocab_size': vocab_size
            }, 'checkpoint_simple_best.pt')

            print(f"  → NEW BEST: {val_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"TF-IDF Baseline: 100.00%")
    print()

    if best_val_acc > 80:
        print("✓✓✓ SUCCESS! Model learned the task!")
        print("    The simplified architecture works much better.")
    elif best_val_acc > 60:
        print("✓ GOOD! Model is learning.")
        print("    Try training a bit longer or increasing model size slightly.")
    else:
        print("⚠️  Still struggling. May need to:")
        print("    - Increase model size (d_model=256)")
        print("    - Train longer")
        print("    - Check for bugs")

    return model, tokenizer, best_val_acc


if __name__ == "__main__":
    model, tokenizer, accuracy = train_simplified_model()
