# modules.py
#
# Additional reusable modules for temporal modeling, uncertainty handling, etc.

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TemporalAttentionModule, self).__init__()
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        Example shape: x is [batch_size, num_nodes, in_dim] across sequential steps,
        or [batch_size, seq_len, feature_dim] for time series.
        Here we keep it simple: treat nodes dimension as "time" dimension.
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Basic self-attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        x_out = torch.matmul(attn_weights, v)
        return x_out

class UncertaintyModule(nn.Module):
    def __init__(self):
        super(UncertaintyModule, self).__init__()
        # Example placeholder to handle aleatoric or epistemic uncertainty
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: input features
        This example simply returns a scaled log variance or similar.
        """
        return x * torch.exp(self.log_var)
