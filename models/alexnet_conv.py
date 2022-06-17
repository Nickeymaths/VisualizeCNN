from typing import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

class AlexnetConv(nn.Module):
    def __init__(self, num_cls=1000):
        self.num_cls = num_cls
        super(AlexnetConv, self).__init__()
        self.init_weights()

        self.conv_layer_indices = [0, 3, 6, 8, 10]
        self.linear_layer_indices = [1, 4, 6]
        
        self.pool_locs = OrderedDict()
        self.feature_maps = OrderedDict()
        self.linear_ft_maps = OrderedDict()

    def init_weights(self):
        alexnet = models.alexnet(pretrained=True)

        self.features = alexnet.features
        self.classifier = alexnet.classifier

        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                self.features[idx] = nn.MaxPool2d(
                    3, stride=2, return_indices=True)

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                self.pool_locs[idx] = location
            else:
                x = layer(x)

        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output
