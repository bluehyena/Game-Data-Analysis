import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
import pickle
import os
from transformers import AutoTokenizer


class NickNameDataset(Dataset):
    def __init__(self, data_type='train'):
        super(NickNameDataset, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
        self.data_type = data_type

        self.file_path = '../data/preprocessing.npz'
        self.data = np.load(self.file_path)
        self.data_length = len(self.data['names'])
        print(self.data_length)

    def get(self, idx):
        name = self.data['names'][idx]
        label = self.data['labels'][idx]

        if label >= 2:
            label = 1

        output = self.tokenizer.encode(name, add_special_tokens=False)
        output = [self.tokenizer.bos_token_id] + output + [self.tokenizer.eos_token_id]

        return output, label

    def len(self):
        return self.data_length