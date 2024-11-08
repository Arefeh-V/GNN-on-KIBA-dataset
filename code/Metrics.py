from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, average_precision_score, mean_squared_error, precision_recall_curve, auc as sklearn_auc
import numpy as np
import matplotlib.pyplot as plt



def compute_ci(preds, labels):
    """Compute Concordance Index (CI)"""
    n = len(labels)
    num_comparable = 0
    num_concordant = 0

    # Compare each pair
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:  # Only compare pairs with different actual labels
                num_comparable += 1
                # Check for concordance
                if (preds[i] > preds[j] and labels[i] > labels[j]) or (preds[i] < preds[j] and labels[i] < labels[j]):
                    num_concordant += 1

    # Compute CI
    ci = num_concordant / num_comparable if num_comparable > 0 else 0
    return ci

def compute_metrics(preds, labels, isTest = False):
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    # print(preds)
    
    preds_binary = (preds > 0.5).astype(int)
    # print(preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_binary, average='binary', zero_division=0)
    auc = roc_auc_score(labels, preds)
    accuracy = accuracy_score(labels, preds_binary)
    mcc = matthews_corrcoef(labels, preds_binary)
    conf_matrix = confusion_matrix(labels, preds_binary)
    average_precision = average_precision_score(labels, preds)
    mse = mean_squared_error(labels, preds)  # MSE computation
    # Compute precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(labels, preds)
    auprc = sklearn_auc(recall_curve, precision_curve)

    ci = None
    # if (isTest):
    #     ci = compute_ci(preds, labels)

    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    # print(classification_report(labels, preds_binary, target_names=['Class 0', 'Class 1'], digits=4, zero_division=0))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "accuracy": accuracy,
        "specificity": specificity,
        "mcc": mcc,
        "conf_matrix": conf_matrix,
        "ap": average_precision,
        "mse": mse,
        "ci": ci,
        "auprc":auprc

    }

def plot_curve(filename, train_aucs=None, val_aucs=None, test_aucs=None):
    plt.figure()
    
     # Plot Train AUC if available
    if train_aucs and len(train_aucs) > 0:
        plt.plot(train_aucs, label='Train AUC', color='green')
    
    # Plot Validation AUC if available
    if val_aucs and len(val_aucs) > 0:
        plt.plot(val_aucs, label='Validation AUC', color='blue')
    
    # Plot Test AUC if available
    if test_aucs and len(test_aucs) > 0:
        plt.plot(test_aucs, label='Test AUC', color='orange')

    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Train, Validation, and Test AUC over Epochs')
    plt.savefig(f'../output_plots_and_metrics/val_test_train_auc_curve_{filename}.png')
    plt.close()

def plot_average_auc(fileName, train_aucs_all, val_aucs_all):
    # Convert lists of lists to numpy arrays for easier manipulation
    train_aucs_all = np.array(train_aucs_all)
    val_aucs_all = np.array(val_aucs_all)
    
    # Compute mean and standard deviation across folds for each epoch
    mean_train_aucs = np.mean(train_aucs_all, axis=0)
    std_train_aucs = np.std(train_aucs_all, axis=0)

    mean_val_aucs = np.mean(val_aucs_all, axis=0)
    std_val_aucs = np.std(val_aucs_all, axis=0)

    epochs = range(1, len(mean_train_aucs) + 1)

    # Plot the average AUC curves with shaded areas for standard deviation
    plt.plot(epochs, mean_train_aucs, label='Average Train AUC', color='blue')
    plt.fill_between(epochs, mean_train_aucs - std_train_aucs, mean_train_aucs + std_train_aucs, color='blue', alpha=0.2)

    plt.plot(epochs, mean_val_aucs, label='Average Validation AUC', color='orange')
    plt.fill_between(epochs, mean_val_aucs - std_val_aucs, mean_val_aucs + std_val_aucs, color='orange', alpha=0.2)

    # Add labels, title, and legend
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('Average AUC Curves Across Folds')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'../output_plots_and_metrics/AverageAUC_{fileName}.png')
    plt.close()