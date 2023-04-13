"""Base Barlow Twins implementation (BarlowTwinsLoss) taken from
https://github.com/facebookresearch/barlowtwins/blob/main/main.py"""
from typing import Tuple
import torch
from torch import nn
from kornia import augmentation as K


class DiffTransform(nn.Module):
    def __init__(self, crop_resize: int = 224):
        super().__init__()

        self.transform = K.AugmentationSequential(
            K.Normalize(mean=torch.tensor(-1), std=torch.tensor(2)),  # from [-1, 1] to [0, 1]
            K.RandomResizedCrop((crop_resize, crop_resize), resample='BICUBIC'),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(1.05, 1.05), p=0.5),
            K.RandomSolarize(thresholds=(0, 0.5), additions=0, p=0.0),
            K.Normalize(mean=torch.tensor(0.5), std=torch.tensor(0.5)),  # back to [-1, 1]
        )
        self.transform_prime = K.AugmentationSequential(
            K.Normalize(mean=torch.tensor(-1), std=torch.tensor(2)),
            K.RandomResizedCrop((crop_resize, crop_resize), resample='BICUBIC'),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(1.05, 1.05), p=0.1),
            K.RandomSolarize(thresholds=(0, 0.5), additions=0, p=0.2),
            K.Normalize(mean=torch.tensor(0.5), std=torch.tensor(0.5)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


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
    # Modify Contrastive Loss Implementation
    # y1, y2 = transform(images)
    # hierarchical_feats1 = discriminator(y1, c=c, step=step, alpha=alpha, return_hierarchical=True)
    # hierarchical_feats2 = discriminator(y2, c=c, step=step, alpha=alpha, return_hierarchical=True)

    feats1 = hierarchical_feats1
    feats2 = hierarchical_feats2
    # hack to compute the cross-corr matrix among gpus on DP
    z1, z2 = barlow_twins(feats1, feats2)  # gather zs from all gpus
    final_loss = barlow_twins.compute_final_loss(z1, z2)  # .module = out of DP

    return final_loss * weighting
    