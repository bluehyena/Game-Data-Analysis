import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
import pickle
import os

class NickNameDataset(Dataset):
    def __init__(self, data_type='train'):
        super(NickNameDataset, self).__init__()

        self.data_type = data_type

        self.file_path = './data/preprocessing.npz'
        self.data = np.load(self.file_path)
        self.data_length = len(self.data['names'])
        print(self.data_length)

    def get(self, idx):
        name = self.data['names'][idx]
        label = self.data['labels'][idx]
        image = self.data['images'][idx]

        if label >= 2:
            label = 1

        return name, image, label

    def len(self):
        return self.data_length