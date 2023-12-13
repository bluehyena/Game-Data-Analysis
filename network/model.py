import torch
import torch.nn as nn
import numpy as np

from layer import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    mask = (seq != pad_idx)
    return mask

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_building=120):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_building, d_hid))

    def _get_sinusoid_encoding_table(self, n_boundary, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_boundary)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
    def __init__(self, n_layer=6, n_head=8, d_model=512, d_inner=2048, d_unit=8, d_street=64,
                 dropout=0.1, use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super(Encoder, self).__init__()

        self.pos_enc = nn.Linear(2, 1)
        self.unit_enc = nn.Linear(d_unit, d_model)
        self.street_enc = nn.Linear(d_street, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout,
                         use_global_attn=use_global_attn, use_street_attn=use_street_attn, use_local_attn=use_local_attn)
            for _ in range(n_layer)
        ])

    def forward(self, src_unit_seq, src_street_seq, global_mask, street_mask, local_mask):
        src_unit_seq = self.pos_enc(src_unit_seq).squeeze(dim=-1)
        src_street_seq = self.pos_enc(src_street_seq).squeeze(dim=-1)

        enc_output = self.unit_enc(src_unit_seq) + self.street_enc(src_street_seq)
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, global_mask, street_mask, local_mask)

        return enc_output

class Decoder(nn.Module):
    def __init__(self, n_layer=6, n_head=8, n_building=120, n_street=50, d_model=512, d_inner=2048, dropout=0.1,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super(Decoder, self).__init__()

        self.type_emb = nn.Embedding(2, d_model)
        self.count_emb = nn.Embedding(n_building + n_street, d_model)
        self.node_enc = nn.Linear(n_building + n_street, d_model)
        self.pos_enc = PositionalEncoding(d_model, n_building=n_building + n_street)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
           DecoderLayer(d_model, d_inner, n_head, dropout,
                        use_global_attn=use_global_attn, use_street_attn=use_street_attn, use_local_attn=use_local_attn)
           for _ in range(n_layer)
        ])
        self.d_model = d_model

    def forward(self, dec_input, enc_output, is_building_tensor, n_building_node, global_mask, street_mask, local_mask, enc_mask):
        dec_output = self.node_enc(dec_input) + self.type_emb(is_building_tensor.long()) + self.pos_enc(dec_input) + self.count_emb(n_building_node)
        dec_output = self.dropout(dec_output)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output, global_mask, street_mask, local_mask, enc_mask)

        return dec_output

class Transformer(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, sos_idx=2, eos_idx=3, pad_idx=4,
                 n_layer=6, n_head=8, dropout=0.1):
        super(Transformer, self).__init__()

        self.sos_idx = sos_idx    # [2, 2, 2, ..., 2]
        self.eos_idx = eos_idx    # [3, 3, 3, ..., 3]
        self.pad_idx = pad_idx    # [4, 4, 4, ..., 4]

        self.encoder = Encoder(n_layer=n_layer, n_head=n_head, d_model=d_model, d_inner=d_inner, dropout=0.1)
        self.decoder = Decoder(n_layer=n_layer, n_head=n_head, _model=d_model, d_inner=d_inner, dropout=0.1,)

        self.dropout = nn.Dropout(dropout)
        self.adj_fc = nn.Linear(d_model, n_building + n_street)

    def forward(self, seq):
        src_mask = get_pad_mask(seq)
        enc_output = self.encoder(seq, src_mask)

        output = self.dropout(dec_output)
        output = self.adj_fc(output)

        return output