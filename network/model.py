import torch
import torch.nn as nn
from transformers import AutoModel

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained('skt/kobert-base-v1')

        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        x = self.model(x)['last_hidden_state'][:, 0]
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x