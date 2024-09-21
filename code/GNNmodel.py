import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, classification_report, precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix, matthews_corrcoef
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv, Linear, to_hetero, BatchNorm
from weightedADJmat import get_new_adj_mat
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
import numpy as np
from typing import Tuple,List

node_types = ['compound', 'protein']
edge_types = [
    ('compound', 'interacts_with', 'protein'), 
    ('protein', 'interacts_with', 'compound')
]

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.01
epochs = 101
hidden_channels = 64





class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.attn_lin = torch.nn.Linear(2 * hidden_channels, 1)
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        # z = torch.cat([z_dict['compound'][edge_label_index[0]], z_dict['protein'][edge_label_index[1]]], dim=-1)
        
        # Compute attention scores
        # attn_scores = F.softmax(self.attn_lin(z), dim=0)
        
        # Apply attention scores to the concatenated embeddings
        # z = attn_scores * z

        # print(z_dict['compound'])
        # z = self.lin1(z)
        # z = F.leaky_relu(z)
        # z = self.lin2(z).sigmoid()
        # return z.view(-1)
        return torch.sigmoid(torch.sum(z_dict['compound'][edge_label_index[0]] * z_dict['protein'][edge_label_index[1]], dim=-1))
        # return torch.sigmoid(-torch.norm(z_dict['compound'][edge_label_index[0]] - z_dict['protein'][edge_label_index[1]], p=2, dim=-1))



class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # Batch normalization
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.bn2 = BatchNorm(out_channels)  # Batch normalization
        self.dropout = dropout
        self.residual = None
        if hidden_channels != out_channels:
            self.residual = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x_res = x
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  # Apply batch normalization
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)  # Apply batch normalization

        if self.residual is not None:
            x_res = self.residual(x_res)
        x += x_res

        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.attn_lin = torch.nn.Linear(2 * hidden_channels, 1)
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        z = torch.cat([z_dict['compound'][edge_label_index[0]], z_dict['protein'][edge_label_index[1]]], dim=-1)
        
        # Compute attention scores
        attn_scores = F.softmax(self.attn_lin(z), dim=0)
        
        # Apply attention scores to the concatenated embeddings
        z = attn_scores * z
        
        z = self.lin1(z).relu()
        z = self.lin2(z).sigmoid()
        return z.view(-1)

class GNNModel(torch.nn.Module):
    def __init__(self, num_compounds, num_proteins, hidden_channels, data):
        super().__init__()
        # Initialize learnable embeddings
        self.compound_embedding = torch.nn.Embedding(num_compounds, hidden_channels)
        self.protein_embedding = torch.nn.Embedding(num_proteins, hidden_channels)
        
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)
        # binary cross entropy loss with logits
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Apply embeddings
        x_dict['compound'] = self.compound_embedding(x_dict['compound'].long())
        x_dict['protein'] = self.protein_embedding(x_dict['protein'].long())

        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def compute_loss(self, z_dict, edge_labels):
        return self.loss(z_dict, edge_labels)



