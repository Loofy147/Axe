import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import AxisMoE, TrajectorySSM

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
