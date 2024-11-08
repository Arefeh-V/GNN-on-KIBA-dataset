import numpy as np
from sklearn.model_selection import KFold
from ModelUtils import runModel, test, compute_metrics
from Metrics import plot_average_auc
import csv


def run_cross_validation(data, train_data, test_data, lr, weight_decay, epochs, optimizer_type, n_splits = 5):
    fileName = f'lr_{lr}_wd_{weight_decay}_epochs_{epochs}_opt_{optimizer_type}'
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    training_edge_index = train_data['compound', 'interacts_with', 'protein'].edge_index.T

    fold = 1
    train_aucs_all = []
    val_aucs_all = []
    trained_model = None

    # Loop over each fold
    for train_idx, val_idx in kf.split(training_edge_index):
        print(f"\nRunning fold {fold}/{n_splits}...")

        # Split the training data into train and validation sets based on the fold
        train_fold_data = train_data.clone()
        val_fold_data = train_data.clone()

        # train_fold_data['compound', 'interacts_with', 'protein'].edge_index = train_data['compound', 'interacts_with', 'protein'].edge_index[:, train_idx]
        # val_fold_data['compound', 'interacts_with', 'protein'].edge_index = train_data['compound', 'interacts_with', 'protein'].edge_index[:, val_idx]

        train_fold_data['compound', 'interacts_with', 'protein'].edge_label_index = train_data['compound', 'interacts_with', 'protein'].edge_label_index[:, train_idx]
        val_fold_data['compound', 'interacts_with', 'protein'].edge_label_index = train_data['compound', 'interacts_with', 'protein'].edge_label_index[:, val_idx]

        train_fold_data['compound', 'interacts_with', 'protein'].edge_label = train_data['compound', 'interacts_with', 'protein'].edge_label[train_idx]
        val_fold_data['compound', 'interacts_with', 'protein'].edge_label = train_data['compound', 'interacts_with', 'protein'].edge_label[val_idx]

        # Train the model on this fold's training data
        # print(train_fold_data)
        # print(val_fold_data)
        model, train_aucs, val_aucs = runModel(data, train_fold_data, val_fold_data, lr, weight_decay, epochs, optimizer_type)

        # Store the results for this fold
        train_aucs_all.append(train_aucs)
        val_aucs_all.append(val_aucs)
        if fold == n_splits:
            print("This is the last fold ... Saving the final model ...")
            trained_model = model

        fold += 1

    print('test evaluation started ... ')
    # After cross-validation, average the results
    avg_val_auc = np.mean([auc[-1] for auc in val_aucs_all])
    print(f"Average Validation AUC after {n_splits}-fold cross-validation: {avg_val_auc:.4f}")
   
    plot_average_auc(fileName, train_aucs_all, val_aucs_all)

    #test on test data
    test_aucs = []
    test_preds, test_labels, test_loss = test(trained_model, test_data)
    test_metrics = compute_metrics(test_preds, test_labels, True)
    test_aucs.append(test_metrics['auc'])

    # Save test metrics to a file (CSV or text)
    with open(f'../output_plots_and_metrics/test_metrics_{fileName}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Metric", "Value"])
        
        # Write each metric and its value
        for metric, value in test_metrics.items():
            writer.writerow([metric, value])

    print("Test metrics saved to csv file.")
    print(f'Test AUC: {test_metrics["auc"]:.2f}')
    print(f'Test Loss: {test_loss:.2f}')

    return avg_val_auc
