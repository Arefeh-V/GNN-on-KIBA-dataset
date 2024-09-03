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
epochs = 100
hidden_channels = 64



def load_data(compound_embeddings_path, protein_embeddings_path, interactions_path):
    # Load embeddings
    compound_embeddings = pd.read_csv(compound_embeddings_path, header=0, index_col=0)
    protein_embeddings = pd.read_csv(protein_embeddings_path, header=0, index_col=0)
    
    # Load interaction data
    interactions = pd.read_csv(interactions_path, header=0)  # this file has columns: 'drug id', 'protein id'
    
    # Combine embeddings
    all_embeddings = pd.concat([compound_embeddings, protein_embeddings])
    
    return compound_embeddings, protein_embeddings, interactions, all_embeddings
    
def create_graph(compound_embeddings, protein_embeddings, interactions, all_embeddings):
    data = HeteroData()
    
    # Add nodes
    data['compound'].x = torch.tensor(compound_embeddings.values, dtype=torch.float)
    data['protein'].x = torch.tensor(protein_embeddings.values, dtype=torch.float)
    
    # Load and normalize the adjacency matrix (new_adj_matrix)
    new_adj_matrix = pd.read_csv('../outputs/KIBA_new_adj_matrix.csv', header=None, index_col=None)
    new_adj_matrix_np = new_adj_matrix.values
    
    # Normalize edge weights to the range [0, 1]
    min_weight = new_adj_matrix_np.min()
    max_weight = new_adj_matrix_np.max()
    edge_weight_np = (new_adj_matrix_np - min_weight) / (max_weight - min_weight)
    
    # Convert to sparse matrix
    edge_weight_sparse = csr_matrix(edge_weight_np)
    
    # Convert the sparse matrix to edge_index and edge_weight
    edge_index, edge_weight = from_scipy_sparse_matrix(edge_weight_sparse)
    
    # Ensure the edge weights are in the correct format
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # Mapping from compound and protein IDs to indices
    compound_id_map = {id_: i for i, id_ in enumerate(compound_embeddings.index)}
    protein_id_map = {id_: i for i, id_ in enumerate(protein_embeddings.index)}
    
    # Add edges with weights
    compound_idx = torch.tensor([compound_id_map[id_] for id_ in interactions['COMPOUND_ID']], dtype=torch.long)
    protein_idx = torch.tensor([protein_id_map[id_] for id_ in interactions['PROTEIN_ID']], dtype=torch.long)
    
    data['compound', 'interacts_with', 'protein'].edge_index = torch.stack([compound_idx, protein_idx], dim=0)
    # data['compound', 'interacts_with', 'protein'].edge_attr = edge_weight  # Assign normalized weights as edge attributes
    
    # For the reverse edge direction (protein -> compound)
    data['protein', 'interacts_with', 'compound'].edge_index = torch.stack([protein_idx, compound_idx], dim=0)
    # data['protein', 'interacts_with', 'compound'].edge_attr = edge_weight  # Use the same weights for reverse edges    
   
    return data

def validate_edge_indices(data):
    # Validate edge indices for compound-protein edges
    compound_protein_edge_index = data['compound', 'interacts_with', 'protein'].edge_index
    max_compound_index = data['compound'].x.size(0)
    max_protein_index = data['protein'].x.size(0)

    assert compound_protein_edge_index[0].max().item() < max_compound_index, \
        f"Compound index {compound_protein_edge_index[0].max().item()} exceeds max index {max_compound_index - 1}"
    assert compound_protein_edge_index[1].max().item() < max_protein_index, \
        f"Protein index {compound_protein_edge_index[1].max().item()} exceeds max index {max_protein_index - 1}"

    # Validate edge indices for protein-compound edges
    protein_compound_edge_index = data['protein', 'interacts_with', 'compound'].edge_index

    assert protein_compound_edge_index[0].max().item() < max_protein_index, \
        f"Protein index {protein_compound_edge_index[0].max().item()} exceeds max index {max_protein_index - 1}"
    assert protein_compound_edge_index[1].max().item() < max_compound_index, \
        f"Compound index {protein_compound_edge_index[1].max().item()} exceeds max index {max_compound_index - 1}"

    print("All edge indices are within bounds.")

def load_and_create_graph(compound_embeddings_path, protein_embeddings_path, interactions_path):
    compound_embeddings, protein_embeddings, interactions, all_embeddings = load_data(
        compound_embeddings_path, protein_embeddings_path, interactions_path
    )
    data = create_graph(compound_embeddings, protein_embeddings, interactions, all_embeddings)

    # Define the transform
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.1, 
            num_test=0.1, 
            is_undirected=True,
            # disjoint_train_ratio=1,
            # neg_sampling_ratio=0.5, 
            # split_labels=True, 
            # add_negative_train_samples=True,
             edge_types=('compound', 'interacts_with', 'protein'),
            rev_edge_types=('protein', 'interacts_with', 'compound'))
    ])
    # Apply the transform
    train_data, val_data, test_data = transform(data)
   
    return data, train_data, val_data, test_data





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
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)
        # binary cross entropy loss with logits
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def compute_loss(self, z_dict, edge_labels):
        return self.loss(z_dict, edge_labels)



def initialize_model(data, node_types, node_features_shape=64):
    # encoder = GNNEncoder(in_channels=64, hidden_channels=64)
    # decoder = EdgeDecoder(hidden_channels=64)
    model = GNNModel(hidden_channels, data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    return model, optimizer

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    z_dict = model(data.x_dict, data.edge_index_dict, data['compound', 'interacts_with', 'protein'].edge_label_index)
    loss = model.compute_loss(z_dict, data['compound', 'interacts_with', 'protein'].edge_label)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test(model, data):
    # with torch.no_grad():
    model.eval()
    preds = model(data.x_dict, data.edge_index_dict, data['compound', 'interacts_with', 'protein'].edge_label_index)
    # preds = model.decoder(z_dict, data['compound', 'interacts_with', 'protein'].edge_label_index)
    return preds, data['compound', 'interacts_with', 'protein'].edge_label


def compute_metrics(preds, labels):
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    preds_binary = (preds > 0).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_binary, average='binary', zero_division=0)
    auc = roc_auc_score(labels, preds)
    accuracy = accuracy_score(labels, preds_binary)
    mcc = matthews_corrcoef(labels, preds_binary)
    conf_matrix = confusion_matrix(labels, preds_binary)
    average_precision = average_precision_score(labels, preds)
    
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    # print(classification_report(labels, preds_binary, target_names=['Class 0', 'Class 1'], digits=4, zero_division=0))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "accuracy": accuracy,
        "specificity": specificity,
        "mcc": mcc,
        "conf_matrix": conf_matrix,
        "ap": average_precision
    }

def final_test(model, test_data): 
    preds, labels = test(model, test_data)
    metrics = compute_metrics(preds, labels)
    # print(f"Precision: {metrics['precision']:.2f}")
    # print(f"Recall: {metrics['recall']:.2f}")
    # print(f"F1 Score: {metrics['f1']:.2f}")
    print(f"AUC: {metrics['auc']:.2f}")
    # print(f"Accuracy: {metrics['accuracy']:.2f}")
    # print(f"Specificity: {metrics['specificity']:.2f}")
    # print(f"MCC: {metrics['mcc']:.2f}")
    print(f"Confusion Matrix: {metrics['conf_matrix']}")
    print(f"average_precision: {metrics['ap']}")

def plot_curve(val_aucs, test_aucs):
    plt.figure()
    plt.plot(val_aucs, label='Validation AUC')
    plt.plot(test_aucs, label='Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Validation and Test AUC over Epochs')
    plt.savefig(f'val_test_auc_curve_lr_{lr}.png')
    plt.close()




def runModel():
    data, train_data, val_data, test_data = load_and_create_graph('../outputs/KIBA_compound_embeddings.csv', '../outputs/KIBA_protein_embeddings.csv', '../dataset/KIBA.csv')
    node_types = [("Compound",2111),("protein",229)]
    model, optimizer = initialize_model(data, node_types)

    val_aucs = []
    test_aucs = []

    for epoch in range(epochs): 
        loss = train(model, optimizer, train_data)
        val_preds, val_labels = test(model, val_data)

        # print('val_preds:')
        # print(val_preds)
        # print('\nval_labels:')
        # print(val_labels)

        val_metrics = compute_metrics(val_preds, val_labels)
        val_aucs.append(val_metrics['auc'])
        
        # test_preds, test_labels = evaluate(model, test_data)
        # test_metrics = compute_metrics(test_preds, test_labels)
        # test_aucs.append(test_metrics['auc'])
        # Test AUC: {test_metrics["auc"]:.4f}

        print(f'Epoch {epoch:>10} | Loss: {loss:.4f} | Val AUC: {val_metrics["auc"]:.4f} | ')




    
if __name__ == "__main__":
    runModel()