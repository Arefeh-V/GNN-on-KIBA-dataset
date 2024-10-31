import pandas as pd
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_scipy_sparse_matrix

from GINmodel import device
from CustomNegSampling import *


def shuffle_edges(data, edge_type=('compound', 'interacts_with', 'protein')):
    edge_index = data[edge_type].edge_index
    num_edges = edge_index.size(1)
    
    # Generate a random permutation of edge indices
    perm = torch.randperm(num_edges)
    
    # Shuffle the edge index according to this permutation
    data[edge_type].edge_index = edge_index[:, perm]
    
    # If you have edge attributes (e.g., weights), shuffle them as well
    if 'edge_attr' in data[edge_type]:
        data[edge_type].edge_attr = data[edge_type].edge_attr[perm]

    return data

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
    
    # num_compound_nodes = len(compound_embeddings)  # Replace with the number of compound nodes
    # num_protein_nodes = len(protein_embeddings)    # Replace with the number of protein nodes

    # # Create one-hot embeddings for compound nodes
    # compound_onehot_embeddings = F.one_hot(torch.arange(num_compound_nodes), num_classes=num_compound_nodes).float()

    # # Create one-hot embeddings for protein nodes
    # protein_onehot_embeddings = F.one_hot(torch.arange(num_protein_nodes), num_classes=num_protein_nodes).float()

    # # Add these one-hot embeddings as node features in the graph
    # data['compound'].x = compound_onehot_embeddings
    # data['protein'].x = protein_onehot_embeddings

    # # Add nodes
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
    edge_weight = edge_weight.clone().detach().float()    

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
    
    # print(data)
    return data

def load_and_create_graph(compound_embeddings_path, protein_embeddings_path, interactions_path):
    compound_embeddings, protein_embeddings, interactions, all_embeddings = load_data(
        compound_embeddings_path, protein_embeddings_path, interactions_path
    )
    data = create_graph(compound_embeddings, protein_embeddings, interactions, all_embeddings)

    # Shuffle the edges before splitting
    data = shuffle_edges(data)

    # Define the transform
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.0, 
            num_test=0.1, 
            is_undirected=True,
            # neg_sampling_ratio=0.0, 
             edge_types=('compound', 'interacts_with', 'protein'),
            # rev_edge_types=('protein', 'interacts_with', 'compound')
            )
    ])
    
    train_data, val_data, test_data = transform(data)
   
   

    num_edges = data['compound', 'interacts_with', 'protein'].edge_index.size(1)
    all_neg_edge_index = generate_neg_samples(data, num_neg_samples=num_edges)
   
   
    # Split negative samples into train and test sets based on the ratio
    train_neg_edges, test_neg_edges = split_neg_samples(all_neg_edge_index, train_ratio=0.9)
    
    
    # Update the train and test sets with their respective negative samples
    train_data = update_data_with_neg_samples(train_data, train_neg_edges)
    test_data = update_data_with_neg_samples(test_data, test_neg_edges)
    
    return data, train_data, val_data, test_data


