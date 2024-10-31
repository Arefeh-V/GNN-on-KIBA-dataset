import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, Linear, to_hetero, BatchNorm

node_types = ['compound', 'protein']
edge_types = [
    ('compound', 'interacts_with', 'protein'), 
    ('protein', 'interacts_with', 'compound')
]

torch.manual_seed(0)    
hidden_channels = 16

# Check if CUDA is available
if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device = torch.device('cuda')
# Check if MPS is available (for Apple Silicon)
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')





class GINEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        
        # GIN Layer 1
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1, train_eps=True)
        self.residual1 = torch.nn.Linear(8, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        
        # GIN Layer 2 (hidden -> hidden)
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2, train_eps=True)
        self.residual2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        # GIN Layer 3 (hidden -> out_channels)
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )
        self.conv3 = GINConv(self.mlp3, train_eps=True)
        self.residual3 = torch.nn.Linear(hidden_channels, out_channels)
        self.bn3 = BatchNorm(out_channels)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_dict, edge_index_dict):
        # First GIN Layer
        x_res1 = x_dict
        x = self.conv1(x_dict, edge_index_dict)
        x = self.dropout(x)
        x_res1 = self.residual1(x_res1)
        x += x_res1
        x = self.bn1(x)
        
        # Second GIN Layer
        x_res2 = x
        x = self.conv2(x, edge_index_dict)
        x = self.dropout(x)
        x_res2 = self.residual2(x_res2)
        # x += x_res1
        x += x_res2
        x = self.bn2(x)
        
        # Third GIN Layer
        x_res3 = x
        x = self.conv3(x, edge_index_dict)
        x = self.dropout(x)
        x_res3 = self.residual3(x_res3)
        # x += x_res2
        x += x_res3
        x = self.bn3(x)
        
        return x


class MLPDecoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(MLPDecoder, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_dict, edge_label_index):
        compound_avg = z_dict['compound'][edge_label_index[0]].mean(dim=-1, keepdim=True)
        protein_avg = z_dict['protein'][edge_label_index[1]].mean(dim=-1, keepdim=True)
        
        z = torch.cat([compound_avg, protein_avg], dim=-1)

        z = F.leaky_relu(z)

        return torch.sigmoid(self.mlp(z)).squeeze(-1)

class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GINEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = MLPDecoder(hidden_channels, hidden_channels)

        # pos_weight=torch.tensor([5.0])
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # x-dict={'cmp':[[...],[...]...],'prt':[[...],[...]...]}
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def compute_loss(self, z_dict, edge_labels):
        return self.loss(z_dict, edge_labels)

class GATEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=8, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv((-1,-1), hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.residual1 = torch.nn.Linear(64, hidden_channels * heads)
        self.bn1 = BatchNorm(hidden_channels * heads)
        
        # Additional GAT layer with 4 heads
        self.conv_mid = GATConv(hidden_channels * heads, hidden_channels, heads=4, concat=True, add_self_loops=False)
        self.residual2 = torch.nn.Linear(hidden_channels * heads, hidden_channels * 4)
        self.bn_mid = BatchNorm(hidden_channels * 4)
        
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, add_self_loops=False)
        self.residual3 = torch.nn.Linear(hidden_channels * heads, out_channels)
        self.bn3 = BatchNorm(out_channels)
        
        self.dropout = dropout

        # Residual connections

        self.feedforward = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x_res1 = x
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_res1 = self.residual1(x_res1)
        x += x_res1
        # x = F.relu(x)
        x = self.bn1(x)

        # x_res2 = x
        # x = self.conv_mid(x, edge_index)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x_res2 = self.residual2(x_res2)
        # x += x_res2
        # x = F.relu(x)
        # x = self.bn_mid(x)

        x = self.conv3(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_res3 = self.residual3(x_res1)
        x += x_res3
        # x = F.relu(x)
        x = self.bn3(x)

        return x
