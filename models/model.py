from .classifier import FracClassifier
from .encoder import UNet3D_Encoder
import torch.nn as nn
import torch
from .losses import CrossEntropyLoss
class Net3D(nn.Module):

    def __init__(
            self,
            projector_base = 16,
            classifier_channels=(256, 128, 64),
            classes=2):
        
        super(Net3D, self).__init__()


        self.encoder = UNet3D_Encoder(base=projector_base, num_classes=21, batch_norm=True, model_location=None, init_weights=False, mode='finetune', pre_task=None)


        self.classifier = FracClassifier(encoder_channels=[self.encoder.out_shapes],
                                         classifier_channels=classifier_channels,
                                         final_channels=classes)

        self.loss = CrossEntropyLoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)
        return x
