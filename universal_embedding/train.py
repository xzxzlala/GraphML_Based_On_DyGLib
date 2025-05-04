import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
from universal_embedding.myData import GraphDataset
from universal_embedding.myModel import UniversalLinkPredModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import StratifiedKFold
import numpy as np


def get_data(data_path, fold_index=0, batch_size=10):
    # Load data
    if data_path is not None and len(data_path) > 3:
        data = pd.read_excel(data_path)
    else:
        # Sample data (replace with your actual data)
        data = {
            'u': [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'v': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            't': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'linked': [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        data = pd.DataFrame(data)

    # Create stratified folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    data['fold'] = -1

    for fold, (_, val_idx) in enumerate(skf.split(data, data['linked'])):
        data.loc[val_idx, 'fold'] = fold

    # Split data
    train_data = data[data['fold'] != fold_index]
    val_data = data[data['fold'] == fold_index]

    # Create datasets
    train_dataset = GraphDataset(train_data, model=None)
    val_dataset = GraphDataset(val_data, model=None)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Only shuffle training data
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        drop_last=False,
        pin_memory=True,
        num_workers=2
    )

    return train_loader, val_loader


def get_model(EMBEDDING_DIM, LEARNING_RATE):
    model = UniversalLinkPredModel(EMBEDDING_DIM)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCELoss()
    return model, optimizer, loss_function


def calculate_metrics(outputs, labels):
    # Convert outputs to binary predictions (0 or 1)
    preds = (outputs > 0.5).float()
    correct = (preds == labels).float().sum()
    accuracy = correct / len(labels)

    # Calculate AUC
    try:
        auc = roc_auc_score(labels.cpu().numpy(), outputs.detach().cpu().numpy())
    except ValueError:
        auc = 0.5  # Default value when only one class present

    return accuracy.item(), auc


def evaluate(model, val_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_auc = 0.0

    with torch.no_grad():
        for batch_idx, (u_emb, v_emb, t, labels) in enumerate(val_loader):
            model = model.to(device)
            u_emb = u_emb.to(device)
            v_emb = v_emb.to(device)
            labels = labels.to(device)

            outputs = model(u_emb, v_emb)
            loss = loss_function(outputs.squeeze(), labels.float())

            acc, auc = calculate_metrics(outputs.squeeze(), labels)
            total_loss += loss.item()
            total_acc += acc
            total_auc += auc

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    avg_auc = total_auc / len(val_loader)

    return avg_loss, avg_acc, avg_auc


def train_epochs(EPOCHS, train_loader, val_loader, model, optimizer, loss_function, device):
    best_val_loss = float('inf')
    model = model.to(device)

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_auc = 0.0

        for batch_idx, (u_emb, v_emb, t, labels) in enumerate(train_loader):
            u_emb = u_emb.to(device)
            v_emb = v_emb.to(device)
            labels=labels.to(device)
            # Forward pass
            outputs = model(u_emb, v_emb)
            loss = loss_function(outputs.squeeze(), labels.float())

            # Calculate metrics
            acc, auc = calculate_metrics(outputs.squeeze(), labels)
            train_acc += acc
            train_auc += auc

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        val_loss, val_acc, val_auc = evaluate(model, val_loader, loss_function, device)

        # Calculate epoch averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_train_auc = train_auc / len(train_loader)

        # Log results
        log_message = (
            f'Epoch [{epoch + 1}/{EPOCHS}]\n'
            f'Train - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}, AUC: {avg_train_auc:.4f}\n'
            f'Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}\n'
        )
        print(log_message)
        with open("./logs.txt", mode="a") as f:
            print(log_message, file=f)

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    # Hyperparameters
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    EMBEDDING_DIM = 128
    BATCH_SIZE = 8
    DATA_PATH = ""
    device = torch.device("cuda")

    # prepare
    model, optimizer, loss_function = get_model(EMBEDDING_DIM, LEARNING_RATE)
    train_loader, val_loader = get_data(DATA_PATH, fold_index=0, batch_size=BATCH_SIZE)

    # train
    train_epochs(EPOCHS, train_loader, val_loader, model, optimizer, loss_function, device)
