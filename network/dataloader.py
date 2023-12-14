import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer


class NickNameDataset(Dataset):
    def __init__(self, data_type='train'):
        super(NickNameDataset, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
        self.data_type = data_type

        self.file_path = '../data/preprocessed.npz'
        self.data = np.load(self.file_path)
        self.data_length = len(self.data['names'])
        print(self.data_length)

    def __getitem__(self, idx):
        name = self.data['names'][idx]
        label = self.data['labels'][idx]

        if label >= 2:
            label = 1

        output = self.tokenizer.encode(name, add_special_tokens=False)
        output = [self.tokenizer.bos_token_id] + output + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (500 - len(output))

        output = torch.tensor(output, dtype=torch.long)
        label = torch.tensor([label], dtype=torch.float32)

        return output, label

    def __len__(self):
        return self.data_length