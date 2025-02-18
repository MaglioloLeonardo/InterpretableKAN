#Credits to: https://github.com/detkov/Convolution-From-Scratch/
import torch
import numpy as np
from typing import List, Tuple, Union


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape
    h_out = np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out,w_out,batch_size,n_channels

def kan_conv2d(matrix: torch.Tensor,
               kernel,
               kernel_side: int,
               stride=(1,1), 
               dilation=(1,1), 
               padding=(0,0),
               device="cuda") -> torch.Tensor:
    h_out, w_out, batch_size, in_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    
    # Estrai tutti i patch da tutti i canali contemporaneamente
    unfold = torch.nn.Unfold((kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride)
    # Risultato: [batch_size, in_channels*kernel_side*kernel_side, h_out*w_out]
    patches = unfold(matrix)

    # Riordina in [N, in_features] = [batch_size*h_out*w_out, in_channels*kernel_side*kernel_side]
    patches = patches.permute(0, 2, 1)  # [batch_size, h_out*w_out, in_features]
    N = batch_size * h_out * w_out
    in_features = in_channels * (kernel_side * kernel_side)
    patches = patches.reshape(N, in_features)

    # Applica il kernel (un singolo filtro produce un singolo canale in uscita)
    out = kernel.forward(patches)  # [N, 1]
    out = out.reshape(batch_size, h_out, w_out).unsqueeze(1)  # [batch_size, 1, h_out, w_out]

    return out


def multiple_convs_kan_conv2d(matrix: torch.Tensor,
                              kernels,
                              kernel_side: int,
                              stride=(1,1),
                              dilation=(1,1),
                              padding=(0,0),
                              device="cuda") -> torch.Tensor:
    batch_size, in_channels, n, m = matrix.shape
    h_out = ((n + 2*padding[0] - dilation[0]*(kernel_side-1) - 1)//stride[0]) + 1
    w_out = ((m + 2*padding[1] - dilation[1]*(kernel_side-1) - 1)//stride[1]) + 1

    # Estrae patch da tutti i canali
    unfold = torch.nn.Unfold((kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride)
    patches = unfold(matrix) # [batch_size, in_channels*kernel_side*kernel_side, h_out*w_out]

    # Riordina in [batch_size * h_out * w_out, in_channels*kernel_side*kernel_side]
    patches = patches.permute(0, 2, 1)
    N = batch_size * h_out * w_out
    in_features = in_channels * (kernel_side*kernel_side)
    patches = patches.reshape(N, in_features)

    # Applica ogni kernel (filtro)
    outputs = []
    for kern in kernels:
        out = kern.conv.forward(patches)  # [N, 1]
        out = out.reshape(batch_size, h_out, w_out).unsqueeze(1)  # [batch_size, 1, h_out, w_out]
        outputs.append(out)

    # Concatena i vari filtri sul canale
    # Se hai n_convs filtri, otterrai [batch_size, n_convs, h_out, w_out]
    matrix_out = torch.cat(outputs, dim=1).to(device)
    return matrix_out



def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix
