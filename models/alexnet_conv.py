from typing import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision


class AlexnetConv(nn.Module):

    #     (features): Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    #     (1): ReLU(inplace=True)
    #     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

    #     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    #     (4): ReLU(inplace=True)
    #     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

    #     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (7): ReLU(inplace=True)
    #     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (9): ReLU(inplace=True)
    #     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (11): ReLU(inplace=True)
    #     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #   )

    #  (classifier): Sequential(
    #     (0): Dropout(p=0.5, inplace=False)
    #     (1): Linear(in_features=9216, out_features=4096, bias=True)
    #     (2): ReLU(inplace=True)
    #     (3): Dropout(p=0.5, inplace=False)
    #     (4): Linear(in_features=4096, out_features=4096, bias=True)
    #     (5): ReLU(inplace=True)
    #     (6): Linear(in_features=4096, out_features=1000, bias=True)
    #   )

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
