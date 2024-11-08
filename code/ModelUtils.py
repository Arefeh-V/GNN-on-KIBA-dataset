import torch
from EarlyStopping import *
from GINmodel import *
from Metrics import *


def initialize_model(data, node_types, lr, weight_decay, optimizer_type):
    
    model = GNNModel(hidden_channels, data).to(device)
    optimizer_instance = None
    if optimizer_type == 'Adam':
        optimizer_instance = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer_instance = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer_instance = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'ASGD':
        optimizer_instance = torch.optim.ASGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer_instance = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Rprop':
        optimizer_instance = torch.optim.Rprop(model.parameters(), lr=lr)

    return model, optimizer_instance


def train(model, optimizer, data):
    model.train()
    z_dict = model(data.x_dict, data.edge_index_dict, data['compound', 'interacts_with', 'protein'].edge_label_index)
    loss = model.compute_loss(z_dict, data['compound', 'interacts_with', 'protein'].edge_label)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    
    return loss.item()


def test(model, data):
    with torch.no_grad():
        model.eval()
        preds = model(data.x_dict, data.edge_index_dict, data['compound', 'interacts_with', 'protein'].edge_label_index)
        labels = data['compound', 'interacts_with', 'protein'].edge_label
        loss = model.compute_loss(preds, labels)

        return preds, labels, loss


def runModel(data, train_data, val_data, lr, weight_decay, epochs, optimizer_type):
    node_types = [("Compound",2111),("protein",229)]
    model, optimizer = initialize_model(data, node_types, lr, weight_decay, optimizer_type)

    train_aucs = []
    val_aucs = []

    early_stopping = EarlyStopping(tolerance=5, min_delta=0.1, counter=0)

    for epoch in range(epochs):
        train_loss = train(model, optimizer, train_data)

        # Compute training AUC
        train_preds, train_labels, _ = test(model, train_data)  # Use the test function to predict on the training set
        train_metrics = compute_metrics(train_preds, train_labels)
        train_aucs.append(train_metrics['auc'])

        #test on validation data
        val_preds, val_labels, val_loss = test(model, val_data)
        val_metrics = compute_metrics(val_preds, val_labels)
        val_aucs.append(val_metrics['auc'])

        
        # early stopping
        early_stopping(train_loss, val_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break

        # plot_curve(f'lr_{lr}_wd_{weight_decay}_epochs_{epochs}_opt_{optimizer_type}', train_aucs, val_aucs)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:>2} | train Loss: {train_loss:.4f} | Val AUC: {val_metrics["auc"]:.4f} | Val AP: {val_metrics["ap"]:.4f} | Val f1: {val_metrics["f1"]:.4f}\n')
        
    return model, train_aucs, val_aucs



    