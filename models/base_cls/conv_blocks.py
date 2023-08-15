import torch
import torch.nn as nn
from torch import cat
import torch.nn.functional as F


def conv_block_3d(in_dim, out_dim, activation, bn=True):
    if bn:
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_dim, affine=True),
            activation)
    else:
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            activation)

def T_conv_block_3d(in_dim, out_dim, bn=True): # NO RELU
    if bn:
        return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2, padding=0),
        nn.InstanceNorm3d(out_dim, affine=True)
        )
    else:
        return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2, padding=0)
        )

def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

def encoder_Layer(in_dim, out_dim, activation, bn=True):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation, bn),
        conv_block_3d(out_dim, out_dim, activation, bn) )

def decoder_Layer(in_dim, out_dim, activation, bn=True):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation, bn),
        conv_block_3d(out_dim, out_dim, activation, bn),
        T_conv_block_3d(out_dim, out_dim//2, bn))