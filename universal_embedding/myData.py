import torch
from torch.utils.data import Dataset
from pandas import DataFrame


class GraphDataset(Dataset):
    def __init__(self, link_table: DataFrame, model):
        """

        :param link_table: dataframe. index, u, v, t, linked
        :param model:
        """
        self.link_table = link_table
        self.model = model

    def get_embedding(self, node, t):
        return torch.rand(128)

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
