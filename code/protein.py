import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np

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

def generate_protein_embeddings(protein_file_path, dimensions=64, workers=10):
    # Read the protein data
    protein_data = pd.read_csv(protein_file_path)
    
    embeddings = {}
    node_sets = [] 

    for index, row in protein_data.iterrows():
        protein_id = row['PROTEIN_ID']
        sequence = row['PROTEIN_SEQUENCE']
        
        # Convert the protein sequence to a graph
        graph = protein_sequence_to_graph(sequence)
        # num_nodes = graph.number_of_nodes()
        # node_names = sorted(list(graph.nodes()))
        # node_sets.append(set(node_names))
        # print(f"Node names for protein {protein_id}: {node_names}")
        
        num_walks, walk_length = get_dynamic_walk_params(graph)
        print('PROTEIN_ID:',row['PROTEIN_ID'], 'num_walks:', num_walks, ' walk_length:', walk_length )
        

        # Initialize Node2Vec model
        node2vec = Node2Vec(graph, p=2, q=0.5, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        
        # Precompute the walks (this step ensures the vocabulary is built)
        walks = node2vec.walks  # This generates the walks
        
        # Fit Node2Vec model
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Get embeddings for all nodes
        default_embedding = np.zeros(dimensions)
        protein_embedding = np.mean([model.wv[node] for node in graph.nodes() if node in model.wv] or [default_embedding], axis=0)
        
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

protein_file_path = '../dataset/KIBA_protein_mapping.csv'
embeddings_df = generate_protein_embeddings(protein_file_path)
embeddings_df.to_csv(f'{mainpath}KIBA_protein_embeddings.csv', index=True)
