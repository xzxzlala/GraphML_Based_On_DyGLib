import logging

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.DyGFormer import DyGFormer
from universal_embedding.myData import GraphDataset
from universal_embedding.myModel import UniversalLinkPredModel
from utils.DataLoader import get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.utils import get_neighbor_sampler
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.sampler import Sampler
import numpy as np


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, pos_fraction=0.5):
        # self.labels = [label for (_, _, _, label) in dataset]
        self.labels=dataset.get_all_labels()
        self.pos_indices = np.where(np.array(self.labels) == 1)[0]
        self.neg_indices = np.where(np.array(self.labels) == 0)[0]
        self.batch_size = batch_size
        self.pos_fraction = pos_fraction
        self.n_pos_per_batch = int(batch_size * pos_fraction)
        self.n_neg_per_batch = batch_size - self.n_pos_per_batch

        # Shuffle indices initially
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)

    def __iter__(self):
        pos_ptr, neg_ptr = 0, 0
        while True:
            # Check if we have enough samples left
            if (pos_ptr + self.n_pos_per_batch > len(self.pos_indices) or
                    neg_ptr + self.n_neg_per_batch > len(self.neg_indices)):
                break

            # Get balanced batch indices
            batch_indices = (
                    list(self.pos_indices[pos_ptr: pos_ptr + self.n_pos_per_batch]) +
                    list(self.neg_indices[neg_ptr: neg_ptr + self.n_neg_per_batch])
            )
            np.random.shuffle(batch_indices)  # Shuffle within batch
            yield batch_indices

            pos_ptr += self.n_pos_per_batch
            neg_ptr += self.n_neg_per_batch

    def __len__(self):
        return min(
            len(self.pos_indices) // self.n_pos_per_batch,
            len(self.neg_indices) // self.n_neg_per_batch
        )


class MLPClassifierForMooc(torch.nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super(MLPClassifierForMooc, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 172)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(172, 1)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_data(data_path, fold_index=0, batch_size=10, device=None):
    # Load pretrained model
    dataset_name = 'mooc'
    model_name = 'DyGFormer'
    seed = 0
    device = 'cpu'

    node_raw_features, edge_raw_features, full_data, _, _, _ = get_node_classification_data(
        dataset_name=dataset_name, val_ratio=0.15, test_ratio=0.15
    )
    neighbor_sampler = get_neighbor_sampler(full_data, sample_neighbor_strategy="recent", time_scaling_factor=1.0,
                                            seed=seed)

    dygformer = DyGFormer(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=100,
        channel_embedding_dim=50,
        patch_size=1,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        max_input_sequence_length=512,
        device=device
    )
    classifier = MLPClassifierForMooc(input_dim=344, dropout=0.1)
    pretrained_model = torch.nn.Sequential(dygformer, classifier)

    dummy_logger = logging.getLogger("dummy")
    dummy_logger.addHandler(logging.NullHandler())

    checkpoint_path = f"./saved_models/{model_name}/{dataset_name}/{model_name}_seed{seed}"
    early_stopping = EarlyStopping(
        patience=0,
        save_model_folder=checkpoint_path,
        save_model_name=f"{model_name}_seed{seed}",
        logger=dummy_logger,
        model_name=model_name
    )
    early_stopping.load_checkpoint(pretrained_model, map_location=device)
    dygformer = pretrained_model[0]
    dygformer.eval()

    # Load data
    data = pd.read_csv(data_path)

    # Create stratified folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    data['fold'] = -1

    for fold, (_, val_idx) in enumerate(skf.split(data, data['linked'])):
        data.loc[val_idx, 'fold'] = fold

    # Split data
    train_data = data[data['fold'] != fold_index]
    val_data = data[data['fold'] == fold_index]

    # Create datasets
    train_dataset = GraphDataset(train_data, model=dygformer)
    val_dataset = GraphDataset(val_data, model=dygformer)

    train_sampler = BalancedBatchSampler(train_dataset, batch_size=batch_size, pos_fraction=0.5)
    val_sampler = BalancedBatchSampler(val_dataset, batch_size=batch_size, pos_fraction=0.5)
    print("Number of positive samples:", len(train_sampler.pos_indices))
    print("Number of negative samples:", len(train_sampler.neg_indices))
    print("Batch size:", train_sampler.batch_size)
    print("Pos per batch:", train_sampler.n_pos_per_batch)
    print("Neg per batch:", train_sampler.n_neg_per_batch)
    print("Estimated number of batches:", len(train_sampler))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        # batch_size=batch_size,
        # shuffle=True,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        sampler=train_sampler  # Apply sampler only to training data
    )

    val_loader = DataLoader(
        val_dataset,
        # batch_size=batch_size,
        # shuffle=False,  # Don't shuffle validation data
        drop_last=False,
        pin_memory=True,
        num_workers=0,
        sampler=val_sampler
        # No sampler for validation data
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

    for epoch in tqdm(range(EPOCHS)):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_auc = 0.0

        for batch_idx, batch in enumerate(train_loader):
            u_emb = torch.stack([item[0][0] for item in batch]).to(device)  # shape [128, emb_dim]
            v_emb = torch.stack([item[1][0] for item in batch]).to(device)  # shape [128, emb_dim]
            labels = torch.tensor([item[3] for item in batch]).to(device)  # shape [128]

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
    EMBEDDING_DIM = 172
    BATCH_SIZE = 128
    DATA_PATH = "./mooc_cleaned.csv"
    device = torch.device("cuda")

    # test data
    # df = pd.read_csv(DATA_PATH)
    # print(df.columns)
    # print(df['linked'].value_counts())

    # prepare
    model, optimizer, loss_function = get_model(EMBEDDING_DIM, LEARNING_RATE)
    train_loader, val_loader = get_data(DATA_PATH, fold_index=0, batch_size=BATCH_SIZE, device=device)

    # # train
    print("start training")
    train_epochs(EPOCHS, train_loader, val_loader, model, optimizer, loss_function, device)
