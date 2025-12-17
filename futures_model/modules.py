import torch
import torch.nn as nn
import torch.nn.functional as F

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
