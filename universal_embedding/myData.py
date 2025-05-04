import torch
from torch.utils.data import Dataset
from pandas import DataFrame
import numpy as np


class GraphDataset(Dataset):
    def __init__(self, link_table: DataFrame, model):
        """

        :param link_table: dataframe. index, u_id, v_id, t, linked  [10, 11, 2000, 1]
        :param model:
        """
        self.link_table = link_table
        self.model = model

    def get_neighbors(self, node, G, t=1000, k=5):
        return [10, 11]

    def get_embedding(self, u: int, timestamp: float, lambda_decay: float = 1.0, neighbors: list = None):
        hu_list = []
        model = self.model
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
        return hu_avg

    def __getitem__(self, index):
        # get u,v,t, linked from link_table
        cur_data = self.link_table.iloc[index]
        # get embedding
        u_embedding = self.get_embedding(cur_data['u'], cur_data['t'])
        v_embedding = self.get_embedding(cur_data['v'], cur_data['t'])
        # get label
        is_linked_label = cur_data['linked']

        return u_embedding, v_embedding, cur_data['t'], is_linked_label

    def __len__(self):
        return len(self.link_table)
