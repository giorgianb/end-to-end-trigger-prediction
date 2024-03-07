"""
This module implements the PyTorch modules that define the
message-passing graph neural networks for hit or segment classification.
"""

import torch
import torch.nn as nn

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh):
        super(EdgeNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, 1))

    # X[batch, N_o, input_dim] Ri/Ro[batch, N_o, N_r]
    def forward(self, X, Ri, Ro):
        # Select the features of the associated nodes
        bo = torch.bmm(Ro.transpose(1, 2), X)   # [batch, N_r, input_dim]
        bi = torch.bmm(Ri.transpose(1, 2), X)   # [batch, N_r, input_dim]
        B = torch.cat([bo, bi], dim=2)  # [batch, N_r, 2*input_dim]
        # Apply the network to each edge
        return self.network(B).squeeze(-1)  # [batch, N_r]

class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh):
        super(NodeNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim*3, output_dim),
            nn.LayerNorm(output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            hidden_activation())

    # X[batch, N_o, input_dim], e[batch, N_e], Ri/Ro[batch, N_o,N_e]
    def forward(self, X, e, Ri, Ro):
        bo = torch.bmm(Ro.transpose(1, 2), X)  # [batch, N_e, N_o]
        bi = torch.bmm(Ri.transpose(1, 2), X)  # [batch, N_e, N_o]
        Rwo = Ro * e[:,None]    # [batch, N_o, N_e]
        Rwi = Ri * e[:,None]    # [batch, N_o, N_e]
        mi = torch.bmm(Rwi, bo) # [batch, N_o]
        mo = torch.bmm(Rwo, bi) # [batch, N_o]
        M = torch.cat([mi, mo, X], dim=2) # [batch, 3
        return self.network(M)

class GNNSegmentClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=2, hidden_dim=8, n_iters=3, hidden_activation=nn.Tanh):
        super(GNNSegmentClassifier, self).__init__()
        self.n_iters = n_iters
        # Setup the input network
        self.input_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            hidden_activation())
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        X, Ri, Ro = inputs         # X[N_o, input_dim]
        # Apply input network to get hidden representation
        H = self.input_network(X)      # H[N_o, hidden_dim]
        # Shortcut connect the inputs onto the hidden representation
        H = torch.cat([H, X], dim=-1)    # H[N_o, hidden_dim+input_dim]
        # Loop over iterations of edge and node networks
        for i in range(self.n_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(H, Ri, Ro))
            # Apply node network
            H = self.node_network(H, e, Ri, Ro)
            # Shortcut connect the inputs onto the hidden representation
            H = torch.cat([H, X], dim=-1)
        # Apply final edge network
        return self.edge_network(H, Ri, Ro)
