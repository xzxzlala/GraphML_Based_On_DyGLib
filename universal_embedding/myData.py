import logging

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from models.DyGFormer import DyGFormer
from utils.DataLoader import get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.utils import get_neighbor_sampler


class GraphDataset(Dataset):
    def __init__(self, link_table: DataFrame, model):
        """

        :param link_table: dataframe. index, u_id, v_id, t, linked  [10, 11, 2000, 1]
        :param model:
        """
        self.device=torch.device("cuda")
        self.link_table = link_table
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.saved_embeddings = {}

    def get_neighbors(self, u: int, t: float, k: int) -> list:
        """
        Find top-k neighbors of node u using time-weighted sigmoid scoring.
        :param u: int, source node ID
        :param G: DataFrame, full interaction dataset with columns ['u', 'v', 't', 'linked']
        :param t: float, current reference time
        :param k: int, number of neighbors to return
        :return: List[int], top-k neighbor node ids
        """
        score_dict = {}
        # df_filtered = self.link_table[(self.link_table['t'] <= t) & (self.link_table['u'] == u)]
        df_filtered = self.link_table[
            (self.link_table['t'] <= t) &
            ((self.link_table['u'] == u) | (self.link_table['v'] == u))
            ]
        # if len(df_filtered)==0:
        #     df_filtered=self.link_table[(self.link_table['u'] == u)]

        for _, row in df_filtered.iterrows():
            v = row['v']
            ts = row['t']
            score = 1 / (1 + np.exp(-(t - ts + 1)))  # sigmoid weight
            score_dict[v] = score_dict.get(v, 0.0) + score

        # Sort and get top-k
        sorted_neighbors = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        topk_neighbors = [v for v, _ in sorted_neighbors[:k]]
        return topk_neighbors

    def get_embedding(self, u: int, timestamp: float, lambda_decay: float = 1.0, neighbors: list = None):
        hu_list = []
        model = self.model
        with torch.no_grad():
            for v in neighbors:
                src_node_ids = np.array([u], dtype=np.int32)
                dst_node_ids = np.array([v], dtype=np.int32)
                node_interact_times = np.array([timestamp], dtype=np.int32)
                hu, _ = model.compute_src_dst_node_temporal_embeddings(src_node_ids, dst_node_ids, node_interact_times)
                hu_list.append(hu.squeeze(0))
        hu_tensor = torch.stack(hu_list, dim=0)  # [k, d]
        positions = torch.arange(len(neighbors)).float()
        weights = torch.softmax(-lambda_decay * positions, dim=0).to(self.device)  # shape: [k]
        hu_avg = torch.sum(weights.unsqueeze(1) * hu_tensor, dim=0)  # shape: [d]
        return hu_avg

    def get_all_labels(self):
        print(self.link_table['linked'].value_counts())
        return self.link_table['linked']

    def __getitem__(self, index):
        if isinstance(index, (list, np.ndarray)):
            return [self.__getitem__(i) for i in index]  #
        # get u,v,t, linked from link_table
        cur_data = self.link_table.iloc[index]
        # get embedding
        if f"{cur_data['u']}_{cur_data['t']}" in self.saved_embeddings:
            u_embedding = self.saved_embeddings[f"{cur_data['u']}_{cur_data['t']}"]
        else:
            u_embedding = self.get_embedding(cur_data['u'], cur_data['t'],
                                             neighbors=self.get_neighbors(cur_data['u'], cur_data['t'], 5))
            self.saved_embeddings[f"{cur_data['u']}_{cur_data['t']}"] = u_embedding
        if f"{cur_data['v']}_{cur_data['t']}" in self.saved_embeddings:
            v_embedding = self.saved_embeddings[f"{cur_data['v']}_{cur_data['t']}"]
        else:
            v_embedding = self.get_embedding(cur_data['v'], cur_data['t'],
                                             neighbors=self.get_neighbors(cur_data['v'], cur_data['t'], 5))
            self.saved_embeddings[f"{cur_data['v']}_{cur_data['t']}"] = v_embedding

        # get label
        is_linked_label = cur_data['linked']

        return u_embedding, v_embedding, cur_data['t'], is_linked_label

    def __len__(self):
        return len(self.link_table)


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


if __name__ == '__main__':

    # Hyperparameters
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    EMBEDDING_DIM = 128
    batch_size = 8
    data_path = "./mooc_cleaned.csv"
    device = torch.device("cuda")

    dataset_name = 'mooc'
    model_name = 'DyGFormer'
    seed = 0
    fold_index = 0

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

    b=train_dataset[0]
    # Dataloder
    batch_size = 32
    num_workers = 4

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集通常不需要打乱顺序
        num_workers=num_workers,
    )
