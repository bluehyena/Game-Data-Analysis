import torch
import torch.nn as nn
from transformers import AutoModel, ViTConfig, ViTModel

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = AutoModel.from_pretrained('skt/kobert-base-v1')
        self.text_model = AutoModel.from_pretrained('sgunderscore/hatescore-korean-hate-speech')

        config = ViTConfig.from_pretrained('google/vit-base-patch16-224', image_size=64)
        self.image_model = ViTModel(config=config)
        # print(self.text_model.apply)
        # print(self.image_model.apply)

        self.fc_1 = nn.Linear(768 * 2, 768 * 2)
        self.fc_2 = nn.Linear(768 * 2, 1)

    def forward(self, x_text, x_image):
        x_text = self.text_model(x_text)['last_hidden_state'][:, 0]
        x_image = self.image_model(x_image)['last_hidden_state'][:, 0]

        x = torch.cat([x_text, x_image], dim=-1)
        x = self.fc_1(x)
        x = self.fc_2(x)

        x = torch.sigmoid(x)

        return x