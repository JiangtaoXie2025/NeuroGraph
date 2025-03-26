# layers.py
#
# Custom layer implementations, such as graph attention or spatial dependency layers.

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.attn = nn.Parameter(torch.empty(out_features, out_features))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, adjacency_matrix):
        """
        x: [batch_size, num_nodes, in_features]
        adjacency_matrix: [batch_size, num_nodes, num_nodes]
        returns: [batch_size, num_nodes, out_features]
        """
        h = self.fc(x)  # transform features
        batch_size, num_nodes, out_dim = h.shape

        # Example naive attention: compute attention logits, apply adjacency
        # shape [batch_size, num_nodes, num_nodes]
        attn_scores = torch.einsum("bni,ij,bnj->bnn", h, self.attn, h)
        attn_scores = attn_scores.masked_fill(adjacency_matrix == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Aggregate neighbors
        x_out = torch.bmm(attn_weights, h)
        return F.elu(x_out)

class SpatialDependencyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SpatialDependencyLayer, self).__init__()
        # Example placeholder, similar concept but possibly with distance or adjacency
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        x: [batch_size, num_nodes, in_features]
        returns: [batch_size, num_nodes, out_features]
        """
        return self.fc(x)
