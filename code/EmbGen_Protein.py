import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
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


def protein_sequence_to_graph(sequence):
    G = nx.DiGraph()
    for i in range(len(sequence) - 1):
        G.add_edge(sequence[i], sequence[i+1])
    return G

def generate_protein_embeddings(protein_file_path, method='node2vec', dimensions=128, workers=10):
    # Read the protein data
    protein_data = pd.read_csv(protein_file_path)
    protein_data.drop_duplicates()

    
    embeddings = {}
    node_sets = [] 

    for index, row in protein_data.iterrows():
        protein_id = row['PROTEIN_ID']
        sequence = row['PROTEIN_SEQUENCE']
        
        # Convert the protein sequence to a graph
        graph = protein_sequence_to_graph(sequence)
      
        num_walks, walk_length = 100,100
        print('PROTEIN_ID:',row['PROTEIN_ID'], 'num_walks:', num_walks, ' walk_length:', walk_length )
        

        if method == 'node2vec':
            # Initialize Node2Vec model
            node2vec = Node2Vec(graph, p=4, q=1, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)

        elif method == 'deepwalk':
            # Initialize DeepWalk model
            model = generate_deepwalk_embeddings(graph, num_walks=num_walks, walk_length=walk_length, dimensions=dimensions)

        
        # Get embeddings for all nodes
        default_embedding = np.zeros(dimensions)
        protein_embedding = np.sum([model.wv[node] for node in graph.nodes() if node in model.wv] or [default_embedding], axis=0)
        
        embeddings[protein_id] = protein_embedding
    
    # Convert embeddings to a DataFrame
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embeddings_df.index.name = 'protein_id'

    # All graphs have the same set of nodes.
    # Node names for protein P42345: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # Maximum number of nodes in a graph: 20
    # max number of edges in a graph: 388
    # min number of edges in a graph: 153

    return embeddings_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python EmbGen_Protein.py <method>")
        sys.exit(1)

    method = sys.argv[1].lower()
    if method not in ['node2vec', 'deepwalk']:
        print("Method must be either 'node2vec' or 'deepwalk'")
        sys.exit(1)

    protein_file_path = '../dataset/Kiba_protein_mapping.csv'
    embeddings_df = generate_protein_embeddings(protein_file_path, method=method)
    embeddings_df.to_csv(f'{mainpath}Kiba_protein_embeddings_{method}.csv', index=True)