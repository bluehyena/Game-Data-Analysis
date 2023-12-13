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
import torch.distributed as dist
from datetime import datetime

import numpy as np
import random
from tqdm import tqdm
import MeCab

from model import get_pad_mask, get_subsequent_mask, get_clipped_adj_matrix
from model import Transformer
from dataloader import NickNameDataset
from test import make_upper_follow_lower_torch_padded

import wandb

class Trainer:
    def __init__(self, batch_size, max_epoch, sos_idx, eos_idx, pad_idx, d_model, n_layer, n_head,
                 dropout, use_checkpoint, checkpoint_epoch, use_tensorboard,
                 val_epoch, save_epoch, lr, local_rank, save_dir_path):
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
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_tensorboard = use_tensorboard
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.local_rank = local_rank
        self.save_dir_path = save_dir_path
        self.lr = lr

        print('local_rank', self.local_rank)

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

        # Only the first dataset initialization will load the full dataset from disk
        self.train_dataset = NickNameDataset(data_type='train')
        self.train_sampler = torch.utils.data.DistributedSampler(dataset=self.train_dataset, rank=rank)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                           sampler=self.train_sampler, num_workers=8, pin_memory=True)

        # Subsequent initializations will use the already loaded full dataset
        self.val_dataset = NickNameDataset(data_type='val')
        self.val_sampler = torch.utils.data.DistributedSampler(dataset=self.val_dataset, rank=rank)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                         sampler=self.val_sampler, num_workers=8, pin_memory=True)

        # Initialize the Transformer model
        self.transformer = Transformer(sos_idx=self.sos_idx, eos_idx=self.eos_idx, pad_idx=self.pad_idx,
                                            d_model=self.d_model,
                                            d_inner=self.d_model * 4, n_layer=self.n_layer, n_head=self.n_head,
                                            dropout=self.dropout).to(device=self.device)
        self.transformer = nn.parallel.DistributedDataParallel(self.transformer, device_ids=[local_rank])

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
        loss = F.binary_cross_entropy(torch.sigmoid(pred[:, :-1]), get_clipped_adj_matrix(trg[:, 1:]), reduction='none')

        # pad_idx에 해당하는 레이블을 무시하기 위한 mask 생성
        pad_mask = get_pad_mask(trg[:, 1:, 0], pad_idx=self.pad_idx).unsqueeze(-1).expand(-1, -1, loss.shape[2])
        sub_mask = get_subsequent_mask(trg[:, :, 0])[:, 1:, :]
        sos_mask = torch.ones_like(sub_mask).to(device=sub_mask.device)
        sos_mask[:, :, 0] = 0
        identity_mask = torch.eye(trg.shape[1]).unsqueeze(0).expand(loss.shape[0], -1, -1).to(device=sub_mask.device)
        identity_mask = 1 - identity_mask[:, 1:, :]
        mask = pad_mask & sub_mask & sos_mask.bool() & identity_mask.bool()

        # mask 적용
        masked_loss = loss * mask.float()
        # 손실의 평균 반환
        return masked_loss.sum() / mask.float().sum()

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
            if self.local_rank == 0:
                wandb.watch(self.transformer.module, log='all')  # <--- 추가된 부분

        for epoch in range(epoch_start, self.max_epoch):
            total_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분

            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the source and target sequences from the batch
                src_unit_seq, src_street_seq, street_index_seq, trg_adj_seq, cur_n_street, cur_n_node = data
                gt_adj_seq = trg_adj_seq.to(device=self.device)
                src_unit_seq = src_unit_seq.to(device=self.device)
                src_street_seq = src_street_seq.to(device=self.device)
                street_index_seq = street_index_seq.to(device=self.device)
                trg_adj_seq = trg_adj_seq.to(device=self.device)
                cur_n_street = cur_n_street.to(device=self.device)
                cur_n_node = cur_n_node.to(device=self.device)

                # Get the model's predictions
                output = self.transformer(src_unit_seq, src_street_seq, street_index_seq, trg_adj_seq,
                                          cur_n_street, cur_n_node)

                # Compute the losses
                loss = self.cross_entropy_loss(output, gt_adj_seq.detach())
                loss_total = loss

                # Backpropagation and optimization step
                loss_total.backward()
                self.optimizer.step()
                self.scheduler.step()

                # 모든 GPU에서 손실 값을 합산 <-- 수정된 부분
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                total_loss += loss

                # 첫 번째 GPU에서만 평균 손실을 계산하고 출력 <-- 수정된 부분
            if self.local_rank == 0:
                loss_mean = total_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss BCE: {loss_mean:.4f}")

                if self.use_tensorboard:
                    wandb.log({"Train bce loss": loss_mean}, step=epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.transformer.module.eval()
                total_val_loss = torch.tensor([0.0], device=self.device)  # <--- 추가된 부분

                with torch.no_grad():
                    # Iterate over batches
                    for data in tqdm(self.val_dataloader):
                        # Get the source and target sequences from the batch
                        src_unit_seq, src_street_seq, street_index_seq, trg_adj_seq, cur_n_street, cur_n_node = data
                        gt_adj_seq = trg_adj_seq.to(device=self.device, dtype=torch.float32)
                        src_unit_seq = src_unit_seq.to(device=self.device, dtype=torch.float32)
                        src_street_seq = src_street_seq.to(device=self.device, dtype=torch.float32)
                        street_index_seq = street_index_seq.to(device=self.device, dtype=torch.long)
                        trg_adj_seq = trg_adj_seq.to(device=self.device, dtype=torch.float32)
                        cur_n_street = cur_n_street.to(device=self.device, dtype=torch.long)
                        cur_n_node = cur_n_node.to(device=self.device)

                        # Greedy Search로 시퀀스 생성
                        decoder_input = trg_adj_seq[:, :cur_n_street[0] + 1]  # 시작 토큰만 포함

                        # output 값을 저장할 텐서를 미리 할당합니다.
                        output_storage = torch.zeros_like(trg_adj_seq, device=self.device)

                        for t in range(cur_n_street[0], gt_adj_seq.shape[1] - 1):  # 임의의 제한값
                            output = self.transformer(src_unit_seq, src_street_seq, street_index_seq, decoder_input,
                                                      cur_n_street, cur_n_node)
                            output_storage[:, t] = output[:, t].detach()
                            next_token = (torch.sigmoid(output) > 0.5).long()[:, t].unsqueeze(-2)
                            decoder_input = torch.cat([decoder_input, next_token], dim=1)
                            decoder_input = make_upper_follow_lower_torch_padded(decoder_input)
                            decoder_input[:, :1] = trg_adj_seq[:, :1]

                        # Compute the losses using the generated sequence
                        loss = self.cross_entropy_loss(output_storage, gt_adj_seq)

                        # 모든 GPU에서 손실 값을 합산 <-- 추가된 부분
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        total_val_loss += loss

                        # 첫 번째 GPU에서만 평균 손실을 계산하고 출력 <-- 추가된 부분
                    if self.local_rank == 0:
                        val_loss_mean = total_val_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss BCE: {val_loss_mean:.4f}")

                        if self.use_tensorboard:
                            wandb.log({"Val bce loss": val_loss_mean}, step=epoch + 1)

                self.transformer.module.train()

            if (epoch + 1) % self.save_epoch == 0:
                # 체크포인트 데이터 준비
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.transformer.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

                if self.local_rank == 0:
                    save_path = os.path.join("./models", self.save_dir_path)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(checkpoint, os.path.join(save_path, "transformer_epoch_" + str(epoch + 1) + ".pth"))


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Maximum number of epochs for training.")
    parser.add_argument("--sos_idx", type=int, default=2, help="Padding index for sequences.")
    parser.add_argument("--eos_idx", type=int, default=3, help="Padding index for sequences.")
    parser.add_argument("--pad_idx", type=int, default=4, help="Padding index for sequences.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--save_dir_path", type=str, default="transformer", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")

    opt = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.save_dir_path = f"{opt.save_dir_path}_{current_time}"

    if opt.local_rank == 0:
        wandb.login(key='5a8475b9b95df52a68ae430b3491fe9f67c327cd')
        wandb.init(project=opt.save_dir_path, config=vars(opt))

        for key, value in wandb.config.items():
            setattr(opt, key, value)

    if opt.local_rank == 0:
        save_path = os.path.join("./models", opt.save_dir_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        config_file_path = os.path.join(save_path, "config.txt")
        with open(config_file_path, "w") as f:
            for arg in vars(opt):
                f.write(f"{arg}: {getattr(opt, arg)}\n")

    if opt.local_rank == 0:
        for arg in vars(opt):
            print(f"{arg}: {getattr(opt, arg)}")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    rank = opt.local_rank
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        if torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') == "cuda:0":
            dist.init_process_group("gloo")

        else:
            dist.init_process_group('nccl')

    # Create a Trainer instance and start the training process
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch, sos_idx=opt.sos_idx, eos_idx=opt.eos_idx, pad_idx=opt.pad_idx,
                      d_street=opt.d_street, d_unit=opt.d_unit, d_model=opt.d_model, n_layer=opt.n_layer, n_head=opt.n_head,
                      n_building=opt.n_building, n_boundary=opt.n_boundary, use_tensorboard=opt.use_tensorboard,
                      dropout=opt.dropout, use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      val_epoch=opt.val_epoch, save_epoch=opt.save_epoch, n_street=opt.n_street, lr=opt.lr,
                      use_global_attn=opt.use_global_attn, use_street_attn=opt.use_street_attn, use_local_attn=opt.use_local_attn,
                      local_rank=opt.local_rank, save_dir_path=opt.save_dir_path)

    trainer.train()