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
    
    def get_neighbors(self, u: int, G: DataFrame, t: float, k: int) -> list:
        """
        Find top-k neighbors of node u using time-weighted sigmoid scoring.
        :param u: int, source node ID
        :param G: DataFrame, full interaction dataset with columns ['u', 'v', 't', 'linked']
        :param t: float, current reference time
        :param k: int, number of neighbors to return
        :return: List[int], top-k neighbor node ids
        """
        score_dict = {}

        df_filtered = G[(G['t'] < t) & (G['u'] == u)]

        for _, row in df_filtered.iterrows():
            v = row['v']
            ts = row['t']
            score = 1 / (1 + np.exp(-(t - ts)))  # sigmoid weight
            score_dict[v] = score_dict.get(v, 0.0) + score

        # Sort and get top-k
        sorted_neighbors = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        topk_neighbors = [v for v, _ in sorted_neighbors[:k]]
        return topk_neighbors


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
