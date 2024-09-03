import torch
import pandas as pd
import numpy as np

def load_features(file_path):
    """
    Load features from a CSV file.

    Args:
    file_path (str): Path to the CSV file

    Returns:
    torch.Tensor: Loaded feature matrix
    """
    df = pd.read_csv(file_path, header=0, index_col=0)
    # Convert all data to numeric, setting errors='coerce' will replace non-numeric values with NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    # Fill NaN values with 0 or any other appropriate value
    df = df.fillna(0)
    return torch.tensor(df.values, dtype=torch.float32)

def save_matrix(matrix, file_path):
    """
    Save a matrix to a CSV file.

    Args:
    matrix (torch.Tensor): Matrix to save
    file_path (str): Path to the CSV file
    """
    np.savetxt(file_path, matrix.numpy(), delimiter=',', fmt='%.2f')

def create_adj_matrix(drug_features_file, target_features_file, interaction_file):
    """
    Create the adjacency matrix from an Excel file.

    Args:
    file_path (str): Path to the Excel file

    Returns:
    torch.Tensor: Created adjacency matrix
    """
    # Load drug and target features to get the list of drugs and targets
    drug_features_df = pd.read_csv(drug_features_file, header=0)
    target_features_df = pd.read_csv(target_features_file, header=0)
    
    drug_ids = drug_features_df['drug_id'].values  # Assuming first column has drug IDs
    target_ids = target_features_df['protein_id'].values  # Assuming first column has target IDs

    # Create a mapping from drug and target IDs to their indices
    drug_to_index = {drug: idx for idx, drug in enumerate(drug_ids)}
    target_to_index = {target: idx for idx, target in enumerate(target_ids)}

    # Load interaction data
    interaction_df = pd.read_csv(interaction_file, header=0)
    adj_matrix = torch.zeros((len(drug_ids), len(target_ids)))

    # print(drug_ids)
    # print(drug_to_index)
    # print(interaction_df[:0, :2])


    for _, row in interaction_df.iterrows():
        if row['COMPOUND_ID'] in drug_to_index and row['PROTEIN_ID'] in target_to_index:
            drug_idx = drug_to_index[row['COMPOUND_ID']]
            target_idx = target_to_index[row['PROTEIN_ID']]
            adj_matrix[drug_idx, target_idx] = 1

    return adj_matrix

def compute_similarity(matrix):
    """
    Compute the cosine similarity matrix for the given feature matrix.

    Args:
    matrix (torch.Tensor): Feature matrix (r x d for drugs or t x d for targets)

    Returns:
    torch.Tensor: Similarity matrix
    """
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)  # Normalize the feature vectors
    similarity_matrix = torch.mm(matrix, matrix.t())
   
    return similarity_matrix

def calculate_new_adj(drug_features, target_features, adj_matrix, threshold=0.5):
    """
    Calculate the new weighted adjacency matrix for a bipartite graph.

    Args:
    drug_features (torch.Tensor): Feature matrix for drugs (r x d)
    target_features (torch.Tensor): Feature matrix for targets (t x d)
    adj_matrix (torch.Tensor): Adjacency matrix (r x t)

    Returns:
    torch.Tensor: New weighted adjacency matrix (r x t)
    """
    # Compute drug-drug and target-target similarity matrices
    sim_rr = compute_similarity(drug_features)
    sim_tt = compute_similarity(target_features)

    # Multiply sim_rr, adj_matrix, and sim_tt to get the new weighted adjacency matrix
    new_adj_matrix = torch.mm(sim_rr, torch.mm(adj_matrix, sim_tt))

    
    

    return new_adj_matrix

def count_non_zero(matrix):
    return torch.count_nonzero(matrix).item()

def get_new_adj_mat(drug_features_path, target_features_path, adj_matrix_path, old_adj_output_path, new_adj_output_path):
    drug_features = load_features(drug_features_path)
    target_features = load_features(target_features_path)
    adj_matrix = create_adj_matrix(drug_features_path, target_features_path, adj_matrix_path)
   
    save_matrix(adj_matrix, old_adj_output_path)

    new_adj_matrix = calculate_new_adj(drug_features, target_features, adj_matrix)
    
    save_matrix(new_adj_matrix, new_adj_output_path)

    old_non_zero = count_non_zero(adj_matrix)
    new_non_zero = count_non_zero(new_adj_matrix)

    print(f"Old adjacency matrix saved to {old_adj_output_path}")
    print(f"New weighted adjacency matrix saved to {new_adj_output_path}")
    print(f"Number of non-zero cells in the old adjacency matrix: {old_non_zero}")
    print(f"Number of non-zero cells in the new weighted adjacency matrix: {new_non_zero}")
    
    return new_adj_matrix
    


if __name__ == "__main__":
    # Define the file paths
    drug_features_path = '../outputs/KIBA_compound_embeddings.csv'
    target_features_path = '../outputs/KIBA_protein_embeddings.csv'
    adj_matrix_path = '../../data/KIBA/KIBA.csv'
    old_adj_output_path = '../outputs/KIBA_old_adj_matrix.csv'
    new_adj_output_path = '../outputs/KIBA_new_adj_matrix.csv'
    
    # Run the main function
    get_new_adj_mat(drug_features_path, target_features_path, adj_matrix_path, old_adj_output_path, new_adj_output_path)