
from LoadData import load_and_create_graph
from GridSearch import grid_search
from Metrics import *


if __name__ == "__main__":
    
    #val_data is empty as we set num_val = 0 in transformer
    data, train_data, val_data, test_data = load_and_create_graph(
    '../outputs/KIBA_compound_embeddings.csv', 
    '../outputs/KIBA_protein_embeddings.csv', 
    '../dataset/KIBA.csv')

    print('\n')
    print(data)
    print('\n')
    print(train_data)
    print('\n')
    print(val_data)
    print('\n')
    print(test_data)
    print('\n')

    # best_params, best_auc = grid_search(data, train_data, test_data)

    
