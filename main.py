import torch
import numpy as np
import logging

from models.DyGFormer import DyGFormer
from utils.utils import get_neighbor_sampler
from utils.DataLoader import get_node_classification_data
from utils.EarlyStopping import EarlyStopping

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


def get_all_hu_and_avg(model, u: int, neighbors: list, timestamp: float, lambda_decay: float = 1.0):
    """
    对 u 和每个邻居 vi，计算 DyGFormer 表示 hu_i，并用 softmax(位置权重) 进行加权平均。
    :param lambda_decay: float, 控制衰减速度，越大越偏向前面的邻居
    """
    hu_list = []

    for v in neighbors:
        src_node_ids = np.array([u])
        dst_node_ids = np.array([v])
        node_interact_times = np.array([timestamp])
        hu, _ = model.compute_src_dst_node_temporal_embeddings(src_node_ids, dst_node_ids, node_interact_times)
        hu_list.append(hu.squeeze(0))
    hu_tensor = torch.stack(hu_list, dim=0)  # [k, d]
    positions = torch.arange(len(neighbors)).float()
    weights = torch.softmax(-lambda_decay * positions, dim=0)  # shape: [k]
    hu_avg = torch.sum(weights.unsqueeze(1) * hu_tensor, dim=0)  # shape: [d]
    return hu_list, hu_avg

if __name__ == "__main__":
    # 基础参数
    dataset_name = 'mooc'
    model_name = 'DyGFormer'
    seed = 0
    device = 'cpu'

    # 载入数据
    node_raw_features, edge_raw_features, full_data, _, _, _ = get_node_classification_data(
        dataset_name=dataset_name, val_ratio=0.15, test_ratio=0.15
    )
    neighbor_sampler = get_neighbor_sampler(full_data, sample_neighbor_strategy="recent", time_scaling_factor=1.0, seed=seed)

    # 构建模型（保持与训练时完全一致）
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
    model = torch.nn.Sequential(dygformer, classifier)

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
    early_stopping.load_checkpoint(model, map_location=device)
    dygformer = model[0]
    dygformer.eval()

    #TODO: part1
    u = 100
    neighbors = [123, 456, 789]
    timestamp = 1000.0

    hu_list, hu_avg = get_all_hu_and_avg(dygformer, u=u, neighbors=neighbors, timestamp=timestamp)

    print("hu_i")
    for i, h in enumerate(hu_list):
        print(f"hu_{i+1}:", h.detach().cpu().numpy())

    print("\nhu_avg")
    print(hu_avg.detach().cpu().numpy())

    #TODO: part3
