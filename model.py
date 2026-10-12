"""NeuroPpred-MoECL model.

ProtT5 and SS8 are processed by three experts and sample-aware MoE routing.
Training can additionally expose contrastive and SS8-reconstruction heads,
while the default forward output remains compatible with prediction code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, 1))

    def forward(self, x, padding_mask=None):
        scores = self.score(x).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask, 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            # BatchNorm mixes statistics across padded residues and batches;
            # sequence-wise GroupNorm is stable for variable-length peptides.
            nn.GroupNorm(1, channels), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class LocalMotifExpert(nn.Module):
    """Multi-scale depthwise CNN for short local sequence motifs."""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([DepthwiseSeparableConv(channels, k, dropout) for k in (3, 5, 7)])
        self.mix_conv = nn.Conv1d(channels * 3, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        xt = x.transpose(1, 2)
        out = torch.cat([branch(xt) for branch in self.branches], dim=1)
        return self.activation(self.norm(self.mix_conv(out).transpose(1, 2)))


class SelectiveStateSpaceExpert(nn.Module):
    """Lightweight bidirectional selective state-space expert.

    The input-dependent decay gate makes the recurrent state selective rather
    than fixed, while avoiding an additional external SSM dependency.
    """

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(channels, channels * 2)
        self.delta = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.out_proj = nn.Linear(channels * 2, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        decay = self.delta(x)
        valid_mask = None if padding_mask is None else ~padding_mask

        def scan(values, gates, valid=None, reverse=False):
            if reverse:
                values, gates = values.flip(1), gates.flip(1)
                if valid is not None:
                    valid = valid.flip(1)
            state = torch.zeros_like(values[:, 0])
            outputs = []
            value_steps = values.unbind(1)
            gate_steps = gates.unbind(1)
            valid_steps = [None] * len(value_steps) if valid is None else valid.unbind(1)
            for value_t, gate_t, valid_t in zip(value_steps, gate_steps, valid_steps):
                candidate = gate_t * state + (1.0 - gate_t) * value_t
                if valid_t is not None:
                    candidate = torch.where(valid_t.unsqueeze(-1), candidate, state)
                state = candidate
                outputs.append(state)
            output = torch.stack(outputs, dim=1)
            return output.flip(1) if reverse else output

        forward_state = scan(value, decay, valid_mask)
        backward_state = scan(value, decay, valid_mask, reverse=True)
        mixed = self.out_proj(torch.cat([forward_state, backward_state], dim=-1))
        if padding_mask is not None:
            mixed = mixed.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.norm(x + self.dropout(mixed))


class StructureAwareExpert(nn.Module):
    """SS8-conditioned modulation of the ProtT5 representation."""

    def __init__(self, channels, ss_dim=64, dropout=0.1):
        super().__init__()
        self.ss_embedding = nn.Embedding(9, ss_dim)
        self.gate = nn.Sequential(nn.Linear(ss_dim, channels), nn.Sigmoid())
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prot, ss):
        structural_gate = self.gate(self.ss_embedding(ss))
        return self.norm(prot + self.dropout(prot * structural_gate))


class Model(nn.Module):
    """Structure-aware dynamic mixture-of-experts classifier."""

    def __init__(self, protT5_dim=1024, max_len=50, emb_dim=320, n_classes=2, dropout=0.3):
        super().__init__()
        self.max_len = max_len
        self.protT5_block = nn.Sequential(
            nn.Linear(protT5_dim, emb_dim), nn.LayerNorm(emb_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.local_expert = LocalMotifExpert(emb_dim, dropout)
        self.global_expert = SelectiveStateSpaceExpert(emb_dim, dropout)
        self.struct_expert = StructureAwareExpert(emb_dim, dropout=dropout)
        self.router = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb_dim // 2, 3)
        )
        self.router_temperature = nn.Parameter(torch.tensor(1.0))
        self.pool = MaskedAttentionPooling(emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.LayerNorm(emb_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(emb_dim // 2, n_classes)
        )
        # Learn a small class-logit calibration offset instead of forcing the
        # decision boundary to remain exactly symmetric after fusion.
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.zeros(n_classes))
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(emb_dim, 128)
        )
        self.ss8_head = nn.Linear(emb_dim, 8)

    @staticmethod
    def _masked_mean(x, padding_mask):
        if padding_mask is None:
            return x.mean(dim=1)
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, protT5_feat, ss, mask=None, return_routing=False, return_aux=False):
        prot = self.protT5_block(F.dropout(protT5_feat, p=0.2, training=self.training))
        if mask is not None:
            prot = prot.masked_fill(mask.unsqueeze(-1), 0.0)
        local = self.local_expert(prot)
        global_context = self.global_expert(prot, padding_mask=mask)
        structural = self.struct_expert(prot, ss)
        if mask is not None:
            local = local.masked_fill(mask.unsqueeze(-1), 0.0)
            structural = structural.masked_fill(mask.unsqueeze(-1), 0.0)

        route_logits = self.router(self._masked_mean(prot, mask))
        temperature = self.router_temperature.clamp_min(0.25)
        # Dense temperature-controlled routing keeps all experts trainable,
        # while still allowing the model to strongly prefer one expert.
        routing = torch.softmax(route_logits / temperature, dim=-1)
        fused = (routing[:, 0, None, None] * local
                 + routing[:, 1, None, None] * global_context
                 + routing[:, 2, None, None] * structural)
        if mask is not None:
            fused = fused.masked_fill(mask.unsqueeze(-1), 0.0)
        pooled = self.pool(fused, mask)
        logits = self.classifier(pooled)
        logits = logits * self.logit_scale.clamp(0.75, 1.5) + self.logit_bias
        if return_aux:
            return {
                "logits": logits,
                "projection": F.normalize(self.projection_head(pooled), dim=-1),
                "ss8_logits": self.ss8_head(fused),
                "routing": routing,
            }
        return (logits, routing) if return_routing else logits
