import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_neg_samples(data, num_neg_samples):

    global_candidate_neg_edges = []  # Global variable to store negative samples for all drugs

    # 1. Compute similarity matrix for drugs and proteins
    drug_features = data['compound'].x.numpy()
    protein_features = data['protein'].x.numpy()

    drug_similarity = cosine_similarity(drug_features)  # Compute drug similarity
    # protein_similarity = cosine_similarity(protein_features)  # Protein similarity (not directly used in this step)
    
    num_drugs = drug_features.shape[0]
    
    # 2. Loop through each drug
    for i in range(num_drugs):
        # 3. Find the 2 most dissimilar drugs to drug i
        drug_sim_i = drug_similarity[i]
        dissimilar_drug_indices = np.argsort(drug_sim_i)[:2]  # Get indices of two least similar drugs
        
        # 4. Gather proteins connected to these two dissimilar drugs
        dissimilar_proteins = []
        for dis_drug_idx in dissimilar_drug_indices:
            dis_drug_proteins = get_connected_proteins(data, dis_drug_idx)
            dissimilar_proteins.extend(dis_drug_proteins)

        dissimilar_proteins = set(dissimilar_proteins)  # Ensure uniqueness

        # 5. Get proteins connected to drug i
        drug_i_proteins = set(get_connected_proteins(data, i))

        # 6. Get the difference: proteins connected to dissimilar drugs but not to drug i
        candidate_neg_proteins = dissimilar_proteins - drug_i_proteins
  
        # 7. Generate candidate negative samples for drug i
        neg_samples_for_i = [(i, prot) for prot in candidate_neg_proteins]
        
        # Store negative samples for this drug in the global variable
        global_candidate_neg_edges.extend(neg_samples_for_i)
        
    # 8. Shuffle the negative samples to get the desired number
    global_candidate_neg_edges = global_candidate_neg_edges[:num_neg_samples]
    
    # Ensure the negative samples have shape [2, N] where N is num_neg_samples
    neg_edge_index = torch.tensor(global_candidate_neg_edges).T

    return neg_edge_index  # Return as edge_index format



def get_connected_proteins(data, drug_idx):
    """ 
    Helper function to return list of proteins connected to a given drug.
    This assumes a bipartite graph structure between drugs and proteins. 
    """
    edge_index = data['compound', 'interacts_with', 'protein'].edge_index
    connected_proteins = edge_index[1][edge_index[0] == drug_idx].tolist()
    return connected_proteins



def update_data_with_neg_samples(data, neg_edge_index):
    
    """
    This function updates the `HeteroData` object by adding the negative samples
    to the `edge_label` and `edge_label_index`.
    """
    # Get the original edge index and labels
    edge_label_index = data['compound', 'interacts_with', 'protein'].edge_label_index
    edge_label = data['compound', 'interacts_with', 'protein'].edge_label

    # 9. Create negative labels (all zeros)
    neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.long)

    # 10. Concatenate the negative edges with the existing edges
    updated_edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=1)
    updated_edge_label = torch.cat([edge_label, neg_labels], dim=0)

    # Update the data object with the new negative samples
    data['compound', 'interacts_with', 'protein'].edge_label_index = updated_edge_label_index
    data['compound', 'interacts_with', 'protein'].edge_label = updated_edge_label

    return data



def split_neg_samples(neg_edge_index, train_ratio=0.8):
    """
    Split negative samples into train and test sets based on the given ratio.
    
    Parameters:
    - neg_edge_index: The complete set of negative samples (edge_index format).
    - train_ratio: The proportion of negative samples to assign to the train set.

    Returns:
    - train_neg_edges: Negative samples for the train set.
    - test_neg_edges: Negative samples for the test set.
    """
    num_neg_samples = neg_edge_index.size(1)
    num_train_samples = int(train_ratio * num_neg_samples)
    
    # Shuffle the indices and split
    perm = torch.randperm(num_neg_samples)
    train_neg_edges = neg_edge_index[:, perm[:num_train_samples]]
    test_neg_edges = neg_edge_index[:, perm[num_train_samples:]]
    
    return train_neg_edges, test_neg_edges



def custom_negative_sampling(edge_index, num_neg_samples, candidate_edges): 
    # Convert candidate edges to a tensor if not already
    candidate_edges = torch.tensor(candidate_edges, dtype=torch.long).T

    # Choose random indices from the candidate edges
    perm = torch.randperm(candidate_edges.size(1))
    neg_samples = candidate_edges[:, perm[:num_neg_samples]]

    return neg_samples