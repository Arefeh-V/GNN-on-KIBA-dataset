
from LoadData import load_and_create_graph
from GridSearch import grid_search
from Metrics import *


if __name__ == "__main__":
    
    #val_data is empty as we set num_val = 0 in transformer
    data, train_data, val_data, test_data = load_and_create_graph(
    # '../outputs/Davis_compound_embeddings_deepwalk.csv', 
    # '../outputs/Davis_protein_embeddings_deepwalk.csv', 
    # '../dataset/Davis.csv')
    '../outputs/KIBA_compound_embeddings_node2vec.csv', 
    '../outputs/KIBA_protein_embeddings_node2vec.csv', 
    '../dataset/KIBA.csv')

    # print('data')
    # print(data)
    
    # print('\ntrain_data')
    # print(train_data)
    # print(train_data['compound', 'interacts_with', 'protein'].edge_index)
    # print(train_data['compound', 'interacts_with', 'protein'].edge_label_index)
    # print(train_data['compound', 'interacts_with', 'protein'].edge_label)
    # print('\ntest_data')
    # print(test_data)
    # print(test_data['compound', 'interacts_with', 'protein'].edge_index)
    # print(test_data['compound', 'interacts_with', 'protein'].edge_label_index)
    # print(test_data['compound', 'interacts_with', 'protein'].edge_label)

    best_params, best_auc = grid_search(data, train_data, test_data)

    
