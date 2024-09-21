import networkx as nx
import matplotlib.pyplot as plt

def visualize_hetero_graph(data, node_color_map={"compound": "red", "protein": "yellow"}, edge_color_map={"interacts_with": "blue", "inhibits": "red"}):
    """
    Visualizes a heterogeneous graph using networkx and matplotlib.
    
    Parameters:
    - data: Heterogeneous graph object from PyTorch Geometric
    - node_color_map: Dictionary mapping node types to colors (optional)
    - edge_color_map: Dictionary mapping edge types to colors (optional)
    """
    print(data.node_types)

    # Initialize a directed graph for the heterogeneous graph
    G = nx.DiGraph()

    # Add nodes with their types
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        print(num_nodes)
        G.add_nodes_from([(i, {"type": node_type}) for i in range(num_nodes)])

    # Add edges with their types
    for edge_type, (src, dst) in data.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        edges = [(src[i].item(), dst[i].item()) for i in range(src.size(0))]
        G.add_edges_from(edges, edge_type=edge_type)

    # Get positions for nodes using a layout (spring layout)
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes with different colors based on their types
    for node_type, color in node_color_map.items():
        node_list = [n for n, attr in G.nodes(data=True) if attr["type"] == node_type]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=color, label=node_type)

    # Draw edges with different colors based on their types
    for edge_type, color in edge_color_map.items():
        edge_list = [(u, v) for u, v, attr in G.edges(data=True) if attr["edge_type"] == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=color, label=edge_type)

    # Draw node labels (optional)
    nx.draw_networkx_labels(G, pos)

    # Show plot with a legend
    plt.legend(scatterpoints=1)
    plt.show()

# Example usage
# visualize_hetero_graph(data)
