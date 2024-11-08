import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nx
from node2vec import Node2Vec
from DeepWalk import generate_deepwalk_embeddings
import sys

mainpath ='../outputs/'

def get_dynamic_walk_params(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    # Adjust based on graph density
    density = num_edges / (num_nodes * (num_nodes - 1))
    
    base_num_walks = 5
    base_walk_length = 5
    
    # Scale walks based on density
    num_walks = int(base_num_walks * (density * 10))
    walk_length = int(base_walk_length * (density * 10))
    
    return num_walks, walk_length
   

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    G = nx.Graph()

    # Add nodes
    for atom in mol.GetAtoms():
        # print(atom.GetSymbol())
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())

    # Add edges
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=bond.GetBondType())

    return G

def generate_embeddings(graph, dimensions, method='node2vec', workers=10):

    # num_walks, walk_length = get_dynamic_walk_params(graph)
    num_walks, walk_length = 200, 100

    if method == 'node2vec':
        # Initialize Node2Vec model
        node2vec = Node2Vec(graph, p=4, q=1, dimensions=dimensions, walk_length=walk_length, num_walks=100, workers=workers)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

    elif method == 'deepwalk':
        # Initialize DeepWalk model
        model = generate_deepwalk_embeddings(graph, num_walks=num_walks, walk_length=walk_length, dimensions=dimensions)

    
    # Get embeddings for all nodes
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    
    # Aggregate node embeddings to get a graph-level embedding
    graph_embedding = np.sum(list(embeddings.values()), axis=0)
    
    return graph_embedding

def generate_drug_embeddings(file_path, dimensions=128):
    # Read the CSV file
    df = pd.read_csv(file_path, header=0)
    df.drop_duplicates()

    smiles_list = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # Assuming the first column is drug_id and the second is SMILES
   
    drug_embeddings = {}
    node_sets = []
    for drug_id, smiles in smiles_list.items():
        try:
            graph = smiles_to_graph(smiles)

            num_nodes = graph.number_of_nodes()
            
            node_names = sorted(list(graph.nodes()))
            node_sets.append(set(node_names))
            # print(f"Node names for compound {drug_id}: {node_names}")

            embedding = generate_embeddings(graph, dimensions=dimensions)
            drug_embeddings[drug_id] = embedding
            print(drug_id)
        except Exception as e:
            print(f"Error processing drug {drug_id}: {e}")
            drug_embeddings[drug_id] = np.zeros(dimensions)
    
    # Convert embeddings to a DataFrame
    embeddings_df = pd.DataFrame.from_dict(drug_embeddings, orient='index')
    embeddings_df.index.name = 'drug_id'
    
    # The graphs have different sets of nodes.
    # Minimum number of nodes in a graph: CHEMBL1972934 :: 10
    # Maximum number of nodes in a graph: CHEMBL409397 :: 268

    return embeddings_df



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python EmbGen_Compound.py <method>")
        sys.exit(1)

    method = sys.argv[1].lower()
    if method not in ['node2vec', 'deepwalk']:
        print("Method must be either 'node2vec' or 'deepwalk'")
        sys.exit(1)

    file_path = '../dataset/Kiba_compound_mapping.csv'
    embeddings_df = generate_drug_embeddings(file_path)
    embeddings_df.to_csv(f'{mainpath}Kiba_compound_embeddings_{method}.csv', index=True)
