import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer


class NickNameDataset(Dataset):
    def __init__(self, data_type='train'):
        super(NickNameDataset, self).__init__()

        # self.tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
        self.tokenizer = AutoTokenizer.from_pretrained('sgunderscore/hatescore-korean-hate-speech')
        self.data_type = data_type

        if data_type == 'train':
            self.file_path = '../data/train_preprocessed.npz'
        else:
            self.file_path = '../data/val_preprocessed.npz'
        self.data = np.load(self.file_path)
        self.data_length = len(self.data['names'])
        print(self.data_length)

    def __getitem__(self, idx):
        name = self.data['names'][idx]
        label = self.data['labels'][idx]
        image = self.data['images'][idx]

        output = self.tokenizer.encode(name, add_special_tokens=False)
        # output = [self.tokenizer.bos_token_id] + output + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (48 - len(output))
        output = output + [self.tokenizer.pad_token_id] * (48 - len(output))

        output = torch.tensor(output, dtype=torch.long)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor([label], dtype=torch.float32)

        image = image.permute(2, 0, 1)

        return output, image, label, name

    def __len__(self):
        return self.data_length