import torch
import torch.nn as nn
import torch.nn.functional as F

from kan_convolutional.KANLinear import KANLinear
import kan_convolutional.convolution
from kan_convolutional.KANConv import KAN_Convolutional_Layer

class LeNet5_KAN(nn.Module):
    def __init__(self, num_classes=62):  # EMNIST Balanced ha 62 classi
        super(LeNet5_KAN, self).__init__()
        
        # Primo strato conv: input=1 canale, output=6 filtri, kernel=5x5
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=1,
            out_channels=6,
            kernel_size=(5,5),
            stride=(1,1),
            padding=(0,0),
            dilation=(1,1),
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.ReLU,
            grid_eps=0.02,
            grid_range=(-1, 1)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Secondo strato conv: input=6 canali, output=16 filtri, kernel=5x5
        self.conv2 = KAN_Convolutional_Layer(
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
            stride=(1,1),
            padding=(0,0),
            dilation=(1,1),
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.ReLU,
            grid_eps=0.02,
            grid_range=(-1, 1)
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Dopo conv1+pool1 (28x28 -> conv5x5->24x24 -> pool->12x12)
        # Dopo conv2+pool2 (12x12 -> conv5x5->8x8 -> pool->4x4)
        # 16 canali da 4x4 => 16*4*4=256
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  

    def forward(self, x):
        # Passo 1: conv + pooling
        x = self.conv1(x)
        x = self.pool1(x)
        
        # Passo 2: conv + pooling
        x = self.conv2(x)
        x = self.pool2(x)
        
        # Flatten
        # Flatten
        x = x.contiguous().view(x.size(0), -1)

        
        # Fully Connected Layers con ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output Layer (senza attivazione)
        x = self.fc3(x)
        return x
