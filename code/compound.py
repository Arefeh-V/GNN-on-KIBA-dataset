import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
from node2vec import Node2Vec

mainpath ='../outputs/'

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    G = nx.Graph()

    # Add nodes
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())

    # Add edges
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=bond.GetBondType())

    return G

def generate_node2vec_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4):
    # Initialize Node2Vec model
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    
    # Fit Node2Vec model
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Get embeddings for all nodes
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    
    # Aggregate node embeddings to get a graph-level embedding
    graph_embedding = np.sum(list(embeddings.values()), axis=0)
    
    return graph_embedding

def generate_drug_embeddings(file_path, dimensions=64):
    # Read the CSV file
    df = pd.read_csv(file_path)
    smiles_list = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # Assuming the first column is drug_id and the second is SMILES

    drug_embeddings = {}
    for drug_id, smiles in smiles_list.items():
        try:
            graph = smiles_to_graph(smiles)
            embedding = generate_node2vec_embeddings(graph, dimensions=dimensions)
            drug_embeddings[drug_id] = embedding
            print(drug_id)
        except Exception as e:
            print(f"Error processing drug {drug_id}: {e}")
            drug_embeddings[drug_id] = np.zeros(dimensions)
    
    # Convert embeddings to a DataFrame
    embeddings_df = pd.DataFrame.from_dict(drug_embeddings, orient='index')
    embeddings_df.index.name = 'drug_id'
    
    # Save the DataFrame to a CSV file
    # embeddings_df.to_csv('drug_embeddings.csv')
    
    return embeddings_df

# Example usage
file_path = '../dataset/KIBA/KIBA_compound_mapping.csv'
embeddings_df = generate_drug_embeddings(file_path)
embeddings_df.to_csv(f'{mainpath}KIBA_compound_embeddings.csv', index=True)
