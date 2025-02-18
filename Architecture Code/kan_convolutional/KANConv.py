import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kan_convolutional.KANLinear import KANLinear
import kan_convolutional.convolution

class KAN_Convolutional_Layer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple = (5,5),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order:int = 3,
            scale_noise:float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.ReLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device: str = "cpu"
        ):
        super(KAN_Convolutional_Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # in_features per KANLinear = in_channels * kernel_width * kernel_height
        in_features = in_channels * kernel_size[0] * kernel_size[1]
        out_features = out_channels

        self.kanlinear = KANLinear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=True,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

    def forward(self, x: torch.Tensor):
        # x: [batch_size, in_channels, H, W]
        # Estrazione patch: [batch_size, in_channels*kernel_size*kernel_size, h_out*w_out]
        patches = self.unfold(x) 

        # [batch_size, h_out*w_out, in_channels*kernel_size*kernel_size]
        patches = patches.permute(0, 2, 1)
        N = patches.size(0)*patches.size(1)
        in_features = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        # [N, in_features]
        patches = patches.reshape(N, in_features)

        # Passa i patch vettorizzati a KANLinear
        out = self.kanlinear(patches)  # [N, out_channels]

        # Risistema in [batch_size, out_channels, h_out, w_out]
        # h_out*w_out = patches.size(1) prima del reshape

        batch_size = x.size(0)
        H, W = x.shape[2], x.shape[3]
        kernelH, kernelW = self.kernel_size
        padH, padW = self.padding
        dilH, dilW = self.dilation
        strH, strW = self.stride

        h_out = (H + 2*padH - dilH*(kernelH-1) - 1)//strH + 1
        w_out = (W + 2*padW - dilW*(kernelW-1) - 1)//strW + 1

        out = out.reshape(batch_size, h_out, w_out, self.out_channels)
        out = out.permute(0, 3, 1, 2)  # [batch_size, out_channels, h_out, w_out]

        return out
