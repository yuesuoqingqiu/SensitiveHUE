import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledSelfAttn(nn.Module):
    """
    Scaled Dot Product Attention.

    Inputs:
        - query: :math:`(B, ..., S, F_1)` where B is the batch_size, S is the sequence length, abd
        :math:`F_1` is the feature size.
        - key: :math:`(B, ..., S, F_1)`.
        - value: :math:`(B, ..., S, F_2)` where :math:`F_2` is the feature size.

    Outpus:
        - result_attn: :math:`(B, ..., S, F_2)`.
        - attn_weight: :math:`(B, ..., S, S)`.
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        masked_scores = scores
        if mask is not None:
            masked_scores = masked_scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(masked_scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), F.softmax(scores, dim=-1)


class MultiHeadAttn(nn.Module):
    """
    Args:
        - input_size (int): input feature size.
        - head_num (int): number of head.

    Input:
        - x: :math:`(B, S, F)`.

    Output:
        - y: :math:`(B, S, F)`.
        - weight_attn: :math:`(B, S, S)`.
    """
    def __init__(self, input_size: int, head_num: int, dropout=0.1):
        super(MultiHeadAttn, self).__init__()
        assert input_size % head_num == 0

        self.head_num = head_num
        self.d_k = input_size // head_num

        self.linear_layers = nn.ModuleList(
            nn.Linear(input_size, input_size, bias=False) for _ in range(3)
        )
        self.output_linear = nn.Linear(input_size, input_size)
        self.attention = ScaledSelfAttn()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [layer(x).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
                             for layer, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)

        return self.output_linear(x), attn.mean(dim=1)
