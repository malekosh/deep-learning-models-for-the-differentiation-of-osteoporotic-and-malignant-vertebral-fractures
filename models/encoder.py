import torch
import torch.nn as nn
from torch import cat
import torch.nn.functional as F
from .base_cls.conv_blocks import *

class UNet3D_Encoder(nn.Module):
    def __init__(self, base=16, num_classes=21, batch_norm=True, model_location=None, init_weights=False, mode='finetune', pre_task=None):
        super(UNet3D_Encoder, self).__init__()

        self.base = base
        self.num_classes = num_classes
        self.mode = mode
        self.pretask = pre_task
        self.out_shapes = self.base * 16
        activation = nn.ReLU(inplace=True)
        bn = batch_norm

        # Down sampling
        self.encoder_1 = encoder_Layer(1, self.base, activation, bn=bn)
        self.pool_1 = max_pooling_3d()
        self.encoder_2 = encoder_Layer(self.base, self.base * 2, activation, bn=bn)
        self.pool_2 = max_pooling_3d()
        self.encoder_3 = encoder_Layer(self.base * 2, self.base * 4, activation, bn=bn)
        self.pool_3 = max_pooling_3d()
        self.encoder_4 = encoder_Layer(self.base * 4, self.base * 8, activation, bn=bn)
        self.pool_4 = max_pooling_3d()
        self.encoder_5 = encoder_Layer(self.base * 8, self.base * 16, activation, bn=bn)

        

        if init_weights and model_location == None:
            self._initialize_weights()

    def _initialize_weights(self):
        print('Kaiming Initializing Weights')
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        B,C, D, H, W = x.size()

        #================================= Start Encoder Block
        E1 = self.encoder_1(x)  # -> [1, 1, 128,128,128]
        x = self.pool_1(E1)  # -> [base, 1, 64*3] -> Skip Connection

        E2 = self.encoder_2(x)  # ->  [base*2, 1, 64*3]
        x = self.pool_2(E2)  # ->  [base*2, 1, 32*3] -> Skip Connection

        E3 = self.encoder_3(x)  # -> [base*4, 1, 32*3]
        x = self.pool_3(E3)  # -> [base*4, 1, 16*3] -> Skip Connection

        E4 = self.encoder_4(x)  # -> [base*8, 1, 16*3]
        x = self.pool_4(E4)  # -> [base*8, 1, 8*3] -> Skip Connection

        x = self.encoder_5(x)  # -> [base*16, 1, 8*3]

        
        return x

    def transferWeights(self, checkpoint):
        pre_net = torch.load(checkpoint, map_location='cpu')
        pretrained_keys = list(pre_net['model_state_dict'])

        # IF TRANSFERRING JIGSAW TO FINETUNE
        if self.mode=='finetune':
            if self.pretask == 'jigsaw':
                pretrained_keys = pretrained_keys[:-6]
            else:
                pretrained_keys = pretrained_keys[:-2]

        model_dict = self.state_dict()
        model_keys = pretrained_keys # If Training Inference

        for i, key in enumerate(pretrained_keys):
            print('Transfering from Self-Supervision Key %s to Model Key %s' % (key, model_keys[i]))
            pretrained_value = pre_net['model_state_dict'][key]
            model_dict[model_keys[i]] = pretrained_value


        #torch.cuda.empty_cache()
        self.load_state_dict(model_dict)

    