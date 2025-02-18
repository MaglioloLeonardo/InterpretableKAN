import torch
import torch.nn.functional as F
import math
from kan_convolutional.KANLinear import KANLinear


class KANLinearFullyConnected(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=3,
        spline_order=5,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation = torch.nn.ReLU, #torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinearFullyConnected, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        

        print(grid_size, spline_order)
        print(base_activation)


        # Build the KAN layers
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:-1]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        

        # Add a final linear layer without activation
        #self.final_layer = torch.nn.Linear(layers_hidden[-2], layers_hidden[-1])

        self.layers.append(
            KANLinear(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=torch.nn.Identity,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        )


    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return F.log_softmax(x, dim=1)
 
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

