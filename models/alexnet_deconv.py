import torch
import torch.nn as nn
import torchvision.models as models

import sys

class AlextnetDeconv(nn.Module):
    """
    vgg16 transpose convolution network architecture
    """
    def __init__(self):
        super(AlextnetDeconv, self).__init__()

        self.features = nn.Sequential(
            # Deconv 1
            nn.MaxUnpool2d(3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 3, padding=1),

            # Deconv 2
            nn.MaxUnpool2d(3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 64, 5, padding=2),

            # Deconv 3
            nn.MaxUnpool2d(3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 11, padding=2, stride=4),
        )

        self.conv2deconv_indices = {
                0:12, 3:9, 6:6, 8:4, 10:2
                }

        self.unpool2pool_indices = {
                10:2, 7:5, 0:12
                }

        self.init_weight()

    def init_weight(self):
        alexnet_pretrained = models.alexnet(pretrained=True)

        for idx, layer in enumerate(alexnet_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data
                
        
    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx]\
                (x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x
