from itertools import product
from CrossValidation import run_cross_validation

param_grid = {
    'lr': [1e-05],
    # 'hidden_channels': [32, 64, 128],
    'weight_decay': [1e-05],
    'epochs': [500],
    'optimizer_type': ['Adam']
}

def grid_search(data, train_data, test_data):
    best_params = None
    best_auc = 0
    
    # Create a list of all possible combinations of hyperparameters
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # Perform grid search
    for params in param_combinations:
        param_dict = dict(zip(param_names, params))
        print('//////////// grid search /////////////')
        print(f"Testing parameters: {param_dict}")
        
        # Run cross-validation with the given parameters
        avg_val_auc = run_cross_validation(data, train_data, test_data, **param_dict)
        
        if avg_val_auc > best_auc:
            best_auc = avg_val_auc
            best_params = param_dict

    print('//// FINISHED ////')
    print(f"Best AUC: {best_auc}")
    print(f"Best parameters: {best_params}")
    print('//////////////////')

    return best_params, best_auc

