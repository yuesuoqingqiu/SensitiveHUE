import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str):
        assert mode in ('norm', 'denorm')

        if mode == 'norm':
            return self._normalize(x)
        return self._denormalize(x)

    def _normalize(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        mean, std = x.mean(dim=dim2reduce, keepdim=True), x.std(dim=dim2reduce, keepdim=True)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)

        dim2reduce = tuple(range(1, x.ndim-1))
        mean, std = x.mean(dim=dim2reduce, keepdim=True), x.std(dim=dim2reduce, keepdim=True)
        x = x * std + mean
        return x
