import pandas as pd
import torch
import random
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from GINmodel import device
from CustomNegSampling import *


class ShuffleEdges:
    def __call__(self, data):
        edge_data = data['compound', 'interacts_with', 'protein']
        # Get the number of edges
        num_edges = edge_data.edge_index.size(1)

        # Generate a permutation of indices
        perm = torch.randperm(num_edges)

        # Shuffle edge_index, edge_attr, and edge_label (if they exist) using perm
        edge_data.edge_index = edge_data.edge_index[:, perm]

        if hasattr(edge_data, 'edge_attr') and edge_data.edge_attr is not None:
            edge_data.edge_attr = edge_data.edge_attr[perm]

        if hasattr(edge_data, 'edge_label') and edge_data.edge_label is not None:
            edge_data.edge_label = edge_data.edge_label[perm]

        # data['compound', 'interacts_with', 'protein'] = edge_data

        return data

class StandardizeFeatures(T.BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'x') and data.x is not None:
            mean = data.x.mean(dim=0)
            std = data.x.std(dim=0)
            data.x = (data.x - mean) / (std + 1e-5)  # Avoid division by zero
        return data 



def shuffle_edges(data, edge_type=('compound', 'interacts_with', 'protein')):
    edge_index = data[edge_type].edge_index
    num_edges = edge_index.size(1)
    
    perm = torch.randperm(num_edges)
    
    data[edge_type].edge_index = edge_index[:, perm]
    
    # If you have edge attributes (e.g., weights), shuffle them as well
    if 'edge_attr' in data[edge_type]:
        data[edge_type].edge_attr = data[edge_type].edge_attr[perm]

    return data


def update_data_with_neg_samples(data, neg_edge_index):
    print('>>>> ',neg_edge_index)
    # Get positive edges and create labels for them
    existing_edge_index = data['compound', 'interacts_with', 'protein'].edge_index
    existing_labels = torch.ones(existing_edge_index.size(1), dtype=torch.long)  # Label positive edges with 1

    # Create labels for the negative edges
    neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float)  # Label negative edges with 0

    # Concatenate positive and negative edges
    combined_edge_index = torch.cat([existing_edge_index, neg_edge_index], dim=1)
    combined_edge_labels = torch.cat([existing_labels, neg_labels], dim=0)

    # combined_edge_index = combined_edge_index.long()
    # combined_edge_labels = combined_edge_labels.long()
    
    # Update the data object with the combined edges and labels
    data['compound', 'interacts_with', 'protein'].edge_index = combined_edge_index
    data['compound', 'interacts_with', 'protein'].edge_label = combined_edge_labels

    return data


AFFINITY_THRESHOLD = 11.3   

def load_data(compound_embeddings_path, protein_embeddings_path, interactions_path):
    compound_embeddings = pd.read_csv(compound_embeddings_path, header=0, index_col=0)
    protein_embeddings = pd.read_csv(protein_embeddings_path, header=0, index_col=0)
    interactions = pd.read_csv(interactions_path, header=0)  # this file has columns: 'drug id', 'protein id'
    
    # Separate positive and negative samples based on affinity threshold
    pos_interactions = interactions[interactions['REG_LABEL'] < AFFINITY_THRESHOLD]
    neg_interactions = interactions[interactions['REG_LABEL'] >= AFFINITY_THRESHOLD]
    
    return compound_embeddings, protein_embeddings, pos_interactions, neg_interactions


def create_graph(compound_embeddings, protein_embeddings, pos_interactions, neg_interactions, add_neg_samples=False):
    data = HeteroData()
    # # Add nodes
    data['compound'].x = torch.tensor(compound_embeddings.values, dtype=torch.float)
    data['protein'].x = torch.tensor(protein_embeddings.values, dtype=torch.float) 

    # Mapping from compound and protein IDs to indices
    compound_id_map = {id_: i for i, id_ in enumerate(compound_embeddings.index)}
    protein_id_map = {id_: i for i, id_ in enumerate(protein_embeddings.index)}
    
    # Convert positive and negative interactions to edge indices
    pos_compound_idx = torch.tensor([compound_id_map[id_] for id_ in pos_interactions['COMPOUND_ID']], dtype=torch.long)
    pos_protein_idx = torch.tensor([protein_id_map[id_] for id_ in pos_interactions['PROTEIN_ID']], dtype=torch.long)
    pos_edge_attributes = torch.tensor(pos_interactions['REG_LABEL'].values, dtype=torch.float)
    
    neg_compound_idx = torch.tensor([compound_id_map[id_] for id_ in neg_interactions['COMPOUND_ID']], dtype=torch.long)
    neg_protein_idx = torch.tensor([protein_id_map[id_] for id_ in neg_interactions['PROTEIN_ID']], dtype=torch.long)
    neg_edge_attributes = torch.tensor(neg_interactions['REG_LABEL'].values, dtype=torch.float)
    
    # Randomly sample negative interactions to match the number of positive interactions
    num_pos_samples = pos_compound_idx.size(0)
    if len(neg_compound_idx) > num_pos_samples:
        # Randomly sample the same number of negative samples as positive samples
        neg_sample_indices = random.sample(range(len(neg_compound_idx)), num_pos_samples)
        neg_compound_idx = neg_compound_idx[neg_sample_indices]
        neg_protein_idx = neg_protein_idx[neg_sample_indices]
        neg_edge_attributes = neg_edge_attributes[neg_sample_indices]

    # Add positive and negative edges with labels
    pos_edge_index = torch.stack([pos_compound_idx, pos_protein_idx], dim=0)
    pos_edge_label = torch.ones(pos_compound_idx.size(0), dtype=torch.float)
    
    neg_edge_index = torch.stack([neg_compound_idx, neg_protein_idx], dim=0)
    neg_edge_label = torch.zeros(neg_compound_idx.size(0), dtype=torch.float)
    
    # Stack edge indices and attributes for positive and negative interactions
    pos_edge_index = torch.stack([pos_compound_idx, pos_protein_idx], dim=0)
    neg_edge_index = torch.stack([neg_compound_idx, neg_protein_idx], dim=0)
    
    # Labels for positive and negative interactions
    pos_edge_label = torch.ones(pos_compound_idx.size(0), dtype=torch.float)
    neg_edge_label = torch.zeros(neg_compound_idx.size(0), dtype=torch.float)
    
    # Combine edge indices, labels, and attributes for all edges
    combined_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    combined_edge_labels = torch.cat([pos_edge_label, neg_edge_label], dim=0)
    combined_edge_attributes = torch.cat([pos_edge_attributes, neg_edge_attributes], dim=0)
    
    # Update the data object with edges, labels, and attributes for compound-protein interaction
    data['compound', 'interacts_with', 'protein'].edge_index = combined_edge_index
    data['compound', 'interacts_with', 'protein'].edge_label = combined_edge_labels
    data['compound', 'interacts_with', 'protein'].edge_attr = combined_edge_attributes

    # To make the graph bidirectional, add protein -> compound edges as well
    data['protein', 'interacts_with', 'compound'].edge_index = combined_edge_index.flip(0)
    data['protein', 'interacts_with', 'compound'].edge_label = combined_edge_labels
    data['protein', 'interacts_with', 'compound'].edge_attr = combined_edge_attributes

    if add_neg_samples:
        num_edges = pos_edge_index.size(1) - neg_edge_index.size(1)
        all_neg_edge_index = generate_neg_samples(data, num_neg_samples=num_edges)
        data = update_data_with_neg_samples(data, all_neg_edge_index)
    
    # Shuffle the edges
    # data = shuffle_edges(data, ('compound', 'interacts_with', 'protein'))
    print('len(pos_interactions)', len(pos_edge_label))
    print('len(neg_interactions)', len(neg_edge_label))

    return data


def load_and_create_graph(compound_embeddings_path, protein_embeddings_path, interactions_path):
    # Load data
    compound_embeddings, protein_embeddings, pos_interactions, neg_interactions = load_data(
        compound_embeddings_path, protein_embeddings_path, interactions_path
    )
    
    # Create the HeteroData graph with positive and negative samples
    data = create_graph(compound_embeddings, protein_embeddings, pos_interactions, neg_interactions)
    
    print('before norm.... ', data['compound'])

    # Define the transform
    transform = T.Compose([
        StandardizeFeatures(),
        T.ToDevice(device),
        ShuffleEdges(),
        T.RandomLinkSplit(
            num_val=0.0, 
            num_test=0.1, 
            is_undirected= True,
            add_negative_train_samples = False,
            neg_sampling_ratio=0.0, 
            disjoint_train_ratio=0.6,
            edge_types=('compound', 'interacts_with', 'protein'),
            rev_edge_types=('protein', 'interacts_with', 'compound'),
            )
        ])
    
    print('after norm.... ', data['compound'])


    train_data, val_data, test_data = transform(data)
    
    return data, train_data, val_data, test_data


