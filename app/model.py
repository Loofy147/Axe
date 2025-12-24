import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from collections import Counter
import pickle
from typing import Dict, List, Tuple
import os

# ============================================================================
# TOKENIZER
# ============================================================================

class CustomTokenizer:
    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.idx_to_word = {idx: word for word, idx in vocab_dict.items()}
        self.pad_token_id = vocab_dict['<PAD>']
        self.unk_token_id = vocab_dict['<UNK>']
        self.vocab_size = len(vocab_dict)

    def encode(self, text, max_length=128):
        text = text.lower()
        # Simple tokenization
        text = text.replace(',', ' ,').replace('.', ' .')
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        words = text.split()

        token_ids = [self.vocab_dict.get(w, self.unk_token_id) for w in words]

        if len(token_ids) < max_length:
            token_ids += [self.pad_token_id] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        words = []
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']

        for idx in token_ids:
            word = self.idx_to_word.get(int(idx), '<UNK>')
            if skip_special_tokens and word in special_tokens:
                continue
            words.append(word)

        return ' '.join(words)


def build_vocabulary(dataset_path, vocab_size=5000):
    """Build vocabulary from dataset"""
    print("Building vocabulary...")
    with open(dataset_path) as f:
        data = json.load(f)

    word_counts = Counter()
    for sample in data['samples']:
        text = sample['text'].lower()
        text = text.replace(',', ' ,').replace('.', ' .')
        words = text.split()
        word_counts.update(words)

    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    top_words = [word for word, _ in word_counts.most_common(vocab_size - len(special_tokens))]
    vocabulary = special_tokens + top_words

    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}

    print(f"Vocabulary size: {len(vocab_dict)}")
    return vocab_dict


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class MixtureOfExperts(nn.Module):
    """MoE for handling multi-dimensional futures"""

    def __init__(self, d_model, n_experts=8, expert_dim=256, dropout=0.1):
        super().__init__()
        self.n_experts = n_experts

        # Experts (simple FFNs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_experts)
        ])

        # Gating network
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x):
        # x: (batch, seq, d_model)
        batch_size, seq_len, d_model = x.shape

        # Compute gates
        gate_logits = self.gate(x)  # (batch, seq, n_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Apply experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        # (batch, seq, n_experts, d_model)

        # Weighted combination
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch, seq, n_experts, 1)
        output = (expert_outputs * gate_weights_expanded).sum(dim=2)  # (batch, seq, d_model)

        # Gate statistics for loss
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(dim=-1).mean()
        gate_std = gate_weights.std(dim=-1).mean()

        return output, gate_entropy, gate_std


class TrajectorySSM(nn.Module):
    """State Space Model for temporal trajectories"""

    def __init__(self, d_model, state_dim=64):
        super().__init__()
        self.state_dim = state_dim

        # State matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.D = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

        # Learnable initialization
        self.h0 = nn.Parameter(torch.zeros(1, state_dim))

    def forward(self, x):
        # x: (batch, seq, d_model)
        batch_size, seq_len, d_model = x.shape

        # Initialize state
        h = self.h0.expand(batch_size, -1)  # (batch, state_dim)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)

            # Update state: h_t = Ah_{t-1} + Bx_t
            h = torch.matmul(h, self.A.t()) + torch.matmul(x_t, self.B.t())

            # Output: y_t = Ch_t + Dx_t
            y = torch.matmul(h, self.C.t()) + torch.matmul(x_t, self.D.t())
            outputs.append(y)

        output = torch.stack(outputs, dim=1)  # (batch, seq, d_model)
        return output, h


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation for axis conditioning"""

    def __init__(self, d_model, n_axes=12):
        super().__init__()
        self.gamma = nn.Linear(n_axes, d_model)
        self.beta = nn.Linear(n_axes, d_model)

    def forward(self, x, axis_weights):
        # x: (batch, seq, d_model)
        # axis_weights: (batch, n_axes)

        gamma = self.gamma(axis_weights).unsqueeze(1)  # (batch, 1, d_model)
        beta = self.beta(axis_weights).unsqueeze(1)

        return gamma * x + beta


# ============================================================================
# MAIN MODEL
# ============================================================================

class FuturesModel(nn.Module):
    """Complete MoE + SSM + FiLM model for futures learning"""

    def __init__(
        self,
        vocab_size,
        n_axes=12,
        d_model=256,
        n_head=8,
        n_layers=4,
        n_experts=8,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_axes = n_axes

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(128, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True),
                'moe': MixtureOfExperts(d_model, n_experts=n_experts, dropout=dropout),
                'ssm': TrajectorySSM(d_model),
                'film': FiLMConditioning(d_model, n_axes),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }) for _ in range(n_layers)
        ])

        # Output heads
        self.axis_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_axes)
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, axis_weights=None):
        batch_size, seq_len = tokens.shape

        # Embeddings
        x = self.token_emb(tokens)
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_emb(pos)
        x = self.dropout(x)

        # Track statistics
        gate_entropies = []
        gate_stds = []

        # Transformer layers with MoE, SSM, FiLM
        for layer in self.layers:
            # Self-attention
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)

            # MoE
            moe_out, gate_entropy, gate_std = layer['moe'](x)
            gate_entropies.append(gate_entropy)
            gate_stds.append(gate_std)
            x = layer['norm2'](x + moe_out)

            # SSM (for temporal modeling)
            ssm_out, _ = layer['ssm'](x)
            x = x + ssm_out

            # FiLM conditioning (if axis weights provided)
            if axis_weights is not None:
                x = layer['film'](x, axis_weights)

            x = layer['norm3'](x)

        # Mean pooling for axis classification
        mask = (tokens != 0).float().unsqueeze(-1)
        x_masked = x * mask
        x_pooled = x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Outputs
        axis_logits = self.axis_head(x_pooled)  # (batch, n_axes) - for regression
        lm_logits = self.lm_head(x)  # (batch, seq, vocab_size)

        stats = {
            'gate_entropy': torch.stack(gate_entropies).mean(),
            'gate_std': torch.stack(gate_stds).mean()
        }

        return axis_logits, lm_logits, stats