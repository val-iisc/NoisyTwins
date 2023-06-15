"""Base Barlow Twins implementation (BarlowTwinsLoss) taken from
https://github.com/facebookresearch/barlowtwins/blob/main/main.py

MIT License
Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Tuple
import torch
from torch import nn


class BarlowTwins(nn.Module):
    def __init__(self, num_feats, lambd=5e-3, sizes=(512, 512, 512, 512), use_projector=True):
        super().__init__()
        self.lambd = lambd

        # projector
        if not use_projector:
            self.projector = nn.Identity()
        else:
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(num_feats, affine=False)

    def forward(self, z1, z2):
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        return self.bn(z1), self.bn(z2)

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def compute_final_loss(self, z1, z2):
        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.size(0))

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def compute_contrastive(hierarchical_feats1, hierarchical_feats2, barlow_twins, weighting=0.01):

    feats1 = hierarchical_feats1
    feats2 = hierarchical_feats2
    # hack to compute the cross-corr matrix among gpus on DP
    z1, z2 = barlow_twins(feats1, feats2)  # gather zs from all gpus
    final_loss = barlow_twins.compute_final_loss(z1, z2)  # .module = out of DP

    return final_loss * weighting
    