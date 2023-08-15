import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_cls.model import Model

## TODO: Make classifier dynamically. Take layers, neurons, and classes as inputs and give the model as output

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(LinearBlock, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.linear_block(x)
        return x


class FracClassifier(Model):

    def __init__(
            self,
            encoder_channels,
            classifier_channels=(256, 128, 64),
            # classifier_channels=(64, 32, 16),
            final_channels=2,
            use_batchnorm=True,
    ):
        super(FracClassifier, self).__init__()
        
        # resnet gives 
        self.initial_conv = nn.Conv3d(encoder_channels[0], 128, kernel_size=3,stride=1, padding=1)
        self.bn_init = nn.InstanceNorm3d(128, affine=True)
        self.initial_conv1 = nn.Conv3d(128, 64, kernel_size=3,stride=1, padding=1)
        self.bn_init1 = nn.InstanceNorm3d(64, affine=True)
        self.initial_conv2 = nn.Conv3d(64, 32, kernel_size=3,stride=1, padding=1)
        self.bn_init2 = nn.InstanceNorm3d(32, affine=True)
        self.initial_conv3 = nn.Conv3d(32, 8, kernel_size=3,stride=1, padding=1)
        self.bn_init3 = nn.InstanceNorm3d(8, affine=True)
        self.vector_shape = encoder_channels[0]
        self.layer1 = LinearBlock(int(512), classifier_channels[0])
        self.layer2 = LinearBlock(classifier_channels[0], classifier_channels[1])
        self.layer3 = LinearBlock(classifier_channels[1], classifier_channels[2])
        
        self.final_dense = nn.Linear(classifier_channels[2], final_channels)
        
        

        self.initialize()
    def create_conv(self,size):
        return nn.Conv3d(size,128,1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bn_init(x)
        x = self.initial_conv1(x)
        x = self.bn_init1(x)
        x = self.initial_conv2(x)
        x = self.bn_init2(x)
        x = self.initial_conv3(x)
        x = self.bn_init3(x)
        x = x.view(-1, int(512))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_dense(x)

        return x