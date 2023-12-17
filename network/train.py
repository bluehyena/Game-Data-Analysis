import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from datetime import datetime

import numpy as np
import random
from tqdm import tqdm

from model import Transformer
from dataloader import NickNameDataset

import wandb

class Trainer:
    def __init__(self, batch_size, max_epoch,
                 dropout, use_checkpoint, checkpoint_epoch, use_tensorboard,
                 val_epoch, save_epoch, lr, save_dir_path):
        """
        Initialize the trainer with the specified parameters.

        Args:
        - batch_size (int): Size of each training batch.
        - max_epoch (int): Maximum number of training epochs.
        - pad_idx (int): Padding index for sequences.
        - d_model (int): Dimension of the model.
        - n_layer (int): Number of transformer layers.
        - n_head (int): Number of multi-head attentions.
        """

        # Initialize trainer parameters
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_tensorboard = use_tensorboard
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.save_dir_path = save_dir_path
        self.lr = lr

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Only the first dataset initialization will load the full dataset from disk
        self.train_dataset = NickNameDataset(data_type='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Subsequent initializations will use the already loaded full dataset
        self.val_dataset = NickNameDataset(data_type='val')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the Transformer model
        self.transformer = Transformer().to(device=self.device)

        # optimizer
        param_optimizer = list(self.transformer.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False, no_deprecation_warning=True)

        # scheduler
        data_len = len(self.train_dataloader)
        num_train_steps = int(data_len / batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_steps)

    def cross_entropy_loss(self, pred, trg):
        """
        Compute the binary cross-entropy loss between predictions and targets.

        Args:
        - pred (torch.Tensor): Model predictions.
        - trg (torch.Tensor): Ground truth labels.

        Returns:
        - torch.Tensor: Computed BCE loss.
        """

        criterion = nn.BCELoss()
        loss = criterion(pred, trg)

        return loss

    def get_accuracy(self, pred, trg):
        pred = pred > 0.5
        return (trg == pred).sum().item()

    def get_true_accuracy(self, pred, trg):
        pred = pred > 0.5
        return str(pred.sum().item()) + ' / ' + str(trg.sum().item())

    def train(self):
        """Training loop for the transformer model."""
        epoch_start = 0

        if self.use_checkpoint:
            checkpoint = torch.load("./models/transformer_epoch_" + str(self.checkpoint_epoch) + ".pt")
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']

        if self.use_tensorboard:
            self.writer = SummaryWriter()
            wandb.watch(self.transformer, log='all')  # <--- 추가된 부분

        for epoch in range(epoch_start, self.max_epoch):
            loss_sum = 0
            true_sum = 0
            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the source and target sequences from the batch
                text, image, label, _ = data
                text = text.to(device=self.device)
                label = label.to(device=self.device)
                image = image.to(device=self.device)

                # Get the model's predictions
                output = self.transformer(text, image)

                # Compute the losses
                loss = self.cross_entropy_loss(output, label.detach())

                # Backpropagation and optimization step
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_sum += loss.detach()
                true_sum += self.get_accuracy(output, label)

                # 첫 번째 GPU에서만 평균 손실을 계산하고 출력 <-- 수정된 부분
            loss_mean = loss_sum / (len(self.train_dataloader))
            true_mean = true_sum / self.train_dataset.data_length
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss BCE: {loss_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - accuracy: {true_mean:.4f}")

            if self.use_tensorboard:
                wandb.log({"Train bce loss": loss_mean}, step=epoch + 1)
                wandb.log({"Train accuracy": true_mean}, step=epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.transformer.eval()

                with torch.no_grad():
                    loss_sum = 0
                    true_sum = 0

                    # Iterate over batches
                    for data in tqdm(self.val_dataloader):
                        # Get the source and target sequences from the batch
                        text, image, label, _ = data
                        text = text.to(device=self.device)
                        label = label.to(device=self.device)
                        image = image.to(device=self.device)

                        # Get the model's predictions
                        output = self.transformer(text, image)

                        # Compute the losses using the generated sequence
                        loss = self.cross_entropy_loss(output, label)

                        loss_sum += loss.detach()
                        true_sum += self.get_accuracy(output, label)

                    val_loss_mean = loss_sum / (len(self.val_dataloader))
                    val_true_mean = true_sum / self.val_dataset.data_length
                    print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss BCE: {val_loss_mean:.4f}")
                    print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation accuracy: {val_true_mean:.4f}")

                    if self.use_tensorboard:
                        wandb.log({"Val bce loss": val_loss_mean}, step=epoch + 1)
                        wandb.log({"Val accuracy": true_mean}, step=epoch + 1)

                self.transformer.train()

            if (epoch + 1) % self.save_epoch == 0:
                # 체크포인트 데이터 준비
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.transformer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

                save_path = os.path.join("./models", self.save_dir_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(checkpoint, os.path.join(save_path, "transformer_epoch_" + str(epoch + 1) + ".pth"))


if __name__ == '__main__':

    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Maximum number of epochs for training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=200, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_dir_path", type=str, default="transformer", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")

    opt = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.save_dir_path = f"{opt.save_dir_path}_{current_time}"

    wandb.login(key='0f272b4978c0b450c3765b24b8abd024d7799e80')
    wandb.init(project='transformer', entity='ssw03270', config=vars(opt), name=opt.save_dir_path)

    for key, value in wandb.config.items():
        setattr(opt, key, value)

    save_path = os.path.join("./models", opt.save_dir_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config_file_path = os.path.join(save_path, "config.txt")
    with open(config_file_path, "w") as f:
        for arg in vars(opt):
            f.write(f"{arg}: {getattr(opt, arg)}\n")

    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # Create a Trainer instance and start the training process
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch, use_tensorboard=opt.use_tensorboard,
                      dropout=opt.dropout, use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      val_epoch=opt.val_epoch, save_epoch=opt.save_epoch, lr=opt.lr, save_dir_path=opt.save_dir_path)

    trainer.train()

    wandb.finish()