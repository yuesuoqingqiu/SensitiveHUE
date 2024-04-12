import torch
import numpy as np
import torch.nn as nn
from typing import Callable
from sensitive_hue.model import TransformerEncoderLayer
from sensitive_hue.modules.revin import RevIN


class Transformer(nn.Module):
    """
    Inputs:
    - x: :math:`(B, S, F)` where B is the batch size, S is the source sequence length,
        F is the embedding dimension.

    Outputs:
        - rec: :math:`(B, S, F)`.
    """
    def __init__(self, step_num_in: int, f_in: int, dim_model: int, head_num: int, dim_hidden_fc: int,
                 encode_layer_num: int, dropout=0.1, sfr=False):
        super().__init__()
        self.f_in = f_in
        self.sfr = sfr  # statistical feature removal
        if sfr:
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
        if self.sfr:
            x = self.rev_in(x, mode='norm')

        pos_embed = self.pos_embed(torch.arange(x.size(1), device=x.device))
        h = self.in_linear(x) + pos_embed
        for encoder in self.encoder:
            h, _ = encoder(h, h, src_mask=mask)

        rec: torch.Tensor = self.rec_linear(h)
        if self.sfr:
            rec = self.rev_in(rec, mode='denorm')

        log_var_recip = self.sigma_linear(h)
        return rec, log_var_recip

    def __str__(self):
        return Transformer.__name__


class TSMask:
    def __init__(self, mask_ratio: float, mask_val=0., mode='continual'):
        assert mode in ('continual', 'random', 'mid', 'tail', 'none')
        self.mask_ratio = mask_ratio
        self.masked_val = mask_val
        self.masked_idx = None
        self.mode = mode
    
    def mask_none(self, x: torch.Tensor):
        return x
    
    def mask_tail(self, x: torch.Tensor):
        start = max(1, int(x.size(1) * (1 - self.mask_ratio)))
        return self.mask_continual(x, start)
    
    def mask_mid(self, x: torch.Tensor):
        start = max(1, int(x.size(1) * 0.5 * (1 - self.mask_ratio)))
        return self.mask_continual(x, start)

    def mask_continual(self, x: torch.Tensor, start=-1):
        seq_len = x.size(1)
        mask_len = max(1, int(self.mask_ratio * seq_len))

        if start < 0:
            start = np.random.randint(0, seq_len - mask_len + 1)

        self.masked_idx = torch.arange(start, start + mask_len, device=x.device)
        masked_x = x.clone()
        masked_x[:, self.masked_idx] = self.masked_val
        return masked_x
    
    def mask_random(self, x: torch.Tensor):
        x_size = x.size()
        x = x.view(-1)

        mask_len = max(1, int(self.mask_ratio * x.size(0)))
        self.masked_idx = torch.randperm(x.size(0), device=x.device)[:mask_len]

        masked_x = x.clone()
        masked_x[self.masked_idx] = self.masked_val
        return masked_x.view(*x_size())
    
    def val_masked_idx(self, x: torch.Tensor):
        if self.mode == 'random':
            return x.view(-1)[self.masked_idx]
        elif self.mode == 'none':
            return x
        return x[:, self.masked_idx]
    
    def apply(self, func: Callable, *args, **kwargs):
        mask_func = getattr(self, f'mask_{self.mode}')
        args = [mask_func(x) for x in args]
        return func(*args, **kwargs)
    
    def apply_inverse(self, func: Callable, *args, **kwargs):
        if self.mode != 'none' and self.masked_idx is None:
            mask_func = getattr(self, f'mask_{self.mode}')
            mask_func(args[0])
        results = func(*[self.val_masked_idx(x) for x in args], **kwargs)
        self.masked_idx = None
        return results

