# ngan.py
#
# Main implementation of the NeuroGraph Attention Network (NGAN).
# This file typically defines the NGAN class, containing initialization,
# forward pass, and any relevant methods (losses, metrics, etc.).

import torch
import torch.nn as nn

from .layers import GraphAttentionLayer
from .modules import TemporalAttentionModule

class NeuroGraphAttentionNetwork(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        super(NeuroGraphAttentionNetwork, self).__init__()
        # Example: one graph attention layer, one temporal attention module
        self.gat_layer = GraphAttentionLayer(in_features, hidden_dim)
        self.temporal_module = TemporalAttentionModule(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adjacency_matrix):
        """
        x: input node features, shape [batch_size, num_nodes, in_features]
        adjacency_matrix: graph adjacency, shape [batch_size, num_nodes, num_nodes]
        returns: class scores (e.g. [batch_size, num_classes])
        """
        # Graph-level feature extraction
        x = self.gat_layer(x, adjacency_matrix)

        # If your data is sequential, integrate temporal attention
        # for demonstration, treat x as [batch_size, num_nodes, hidden_dim]
        x = self.temporal_module(x)

        # Global average pool or flatten
        x = x.mean(dim=1)  # example: average across nodes

        # Final classification or other task
        logits = self.fc_out(x)
        return logits
