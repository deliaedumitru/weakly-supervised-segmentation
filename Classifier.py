import torch
from torch import nn
import torchvision.transforms.functional as TF

from custom_layers import VGGBlock

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50*50*32, 5),
            nn.Sigmoid(),
#             nn.Linear(128, 5)
        )
        self.double_conv1 = VGGBlock(3, 16)
        self.double_conv2 = VGGBlock(16, 32)
        self.maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.1)
        self.activation_map = None

    def forward(self, x):
        x = self.double_conv1(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.double_conv2(x)
        x = self.dropout(x)
        x = self.activation_map = self.maxpool(x)
        x = self.flatten(x)
        return x