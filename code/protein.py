import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np

mainpath ='../outputs/'

def protein_sequence_to_graph(sequence):
    G = nx.DiGraph()
    for i in range(len(sequence) - 1):
        G.add_edge(sequence[i], sequence[i+1])
    return G

def generate_protein_embeddings(protein_file_path, dimensions=64, walk_length=30, num_walks=200, workers=4):
    # Read the protein data
    protein_data = pd.read_csv(protein_file_path)
    
    embeddings = {}
    
    for index, row in protein_data.iterrows():
        protein_id = row['PROTEIN_ID']
        sequence = row['PROTEIN_SEQUENCE']
        
        # Convert the protein sequence to a graph
        graph = protein_sequence_to_graph(sequence)
        
        # Initialize Node2Vec model
        node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        
        # Fit Node2Vec model
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Get embeddings for all nodes
        default_embedding = np.zeros(dimensions)
        protein_embedding = np.sum([model.wv[node] for node in graph.nodes() if node in model.wv] or [default_embedding], axis=0)
        
        embeddings[protein_id] = protein_embedding
    
    # Convert embeddings to a DataFrame
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embeddings_df.index.name = 'protein_id'
    
    # Save the embeddings to a CSV file
    # embeddings_df.to_csv('protein_embeddings.csv', index=True)
    
    return embeddings_df

# Example usage
protein_file_path = '../dataset/KIBA_protein_mapping.csv'
embeddings_df = generate_protein_embeddings(protein_file_path)
embeddings_df.to_csv(f'{mainpath}KIBA_protein_embeddings.csv', index=True)
