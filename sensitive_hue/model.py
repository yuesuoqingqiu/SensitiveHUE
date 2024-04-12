import torch
import torch.nn as nn
from .modules.attention import MultiHeadAttn
from .modules.revin import RevIN


class TransformerEncoderLayer(nn.Module):
    """
    Inputs:
        - src: :math:`(B, S, F)` where B is the batch size, S is the target sequence length,
          F is the embedding dimension.
        - value: :math:`(B, S, F)`.

    Outputs:
        - attn_output: :math:`(B, S, F)`.
        - attn_output_weights: :math:`(B, S, S)`.
    """
    def __init__(self, d_model: int, head_num: int, dim_hidden_fc: int, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiHeadAttn(d_model, head_num, dropout)

        self.linear_layer = nn.Sequential(
            nn.Linear(d_model, dim_hidden_fc, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden_fc, d_model, bias=False)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, key, src_mask=None):
        src2, attn_weight = self.attn(src, key, key, mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))
        src = self.norm2(src + self.dropout2(self.linear_layer(src)))
        return src, attn_weight


class SensitiveHUE(nn.Module):
    """
    Inputs:
    - x: :math:`(B, S, F)` where B is the batch size, S is the source sequence length,
        F is the embedding dimension.

    Outputs:
        - rec: :math:`(B, S, F)`.
    """
    def __init__(self, step_num_in: int, f_in: int, dim_model: int, head_num: int, dim_hidden_fc: int,
                 encode_layer_num: int, dropout=0.1):
        super().__init__()
        self.f_in = f_in
        self.eps = 1e-5

        self.rev_in = RevIN(f_in, affine=False)
        self.in_linear = nn.Linear(f_in, dim_model)
        self.pos_embed = nn.Embedding(step_num_in, dim_model)

        self.encoder = nn.ModuleList(
            TransformerEncoderLayer(dim_model, head_num, dim_hidden_fc, dropout)
            for _ in range(encode_layer_num)
        )

        self.rec_linear = nn.Linear(dim_model, f_in)
        self.sigma_linear = nn.Linear(dim_model, f_in)

    def forward(self, x: torch.Tensor, mask=None):
        pos_embed = self.pos_embed(torch.arange(x.size(1), device=x.device))
        x = self.rev_in(x, mode='norm')

        h = self.in_linear(x) + pos_embed
        for encoder in self.encoder:
            h, _ = encoder(h, h, src_mask=mask)

        rec: torch.Tensor = self.rec_linear(h)
        rec = self.rev_in(rec, mode='denorm')

        log_var_recip = self.sigma_linear(h)
        return rec, log_var_recip

    def __str__(self):
        return SensitiveHUE.__name__
