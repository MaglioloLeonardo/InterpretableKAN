import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import random
from pytorch_grad_cam import GradCAM
from kan_convolutional.KANLinear import KANLinear
import kan_convolutional.convolution
from kan_convolutional.KANConv import KAN_Convolutional_Layer

class LeNet5_KAN(nn.Module):
    def __init__(self, num_classes=62):
        super(LeNet5_KAN, self).__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=1, out_channels=6, kernel_size=(5,5),
            stride=(1,1), padding=(0,0), dilation=(1,1),
            grid_size=5, spline_order=3,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            base_activation=torch.nn.ReLU,
            grid_eps=0.02, grid_range=(-1,1)
        )
        self.pool1 = nn.AvgPool2d(2,2)
        self.conv2 = KAN_Convolutional_Layer(
            in_channels=6, out_channels=16, kernel_size=(5,5),
            stride=(1,1), padding=(0,0), dilation=(1,1),
            grid_size=5, spline_order=3,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            base_activation=torch.nn.ReLU,
            grid_eps=0.02, grid_range=(-1,1)
        )
        self.pool2 = nn.AvgPool2d(2,2)
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)
        self.last_conv_feat = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        self.last_conv_feat = x
        x = self.pool2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def read_idx_images(path):
    with open(path,'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)

def read_idx_labels(path):
    with open(path,'rb') as f:
        _, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def main():
    seed = 14
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/home/magliolo/.cache/emnist/gzip/'

    # load train+test
    tr_imgs = read_idx_images(os.path.join(data_dir,'emnist-byclass-train-images-idx3-ubyte'))
    tr_lbls = read_idx_labels(os.path.join(data_dir,'emnist-byclass-train-labels-idx1-ubyte'))
    te_imgs = read_idx_images(os.path.join(data_dir,'emnist-byclass-test-images-idx3-ubyte'))
    te_lbls = read_idx_labels(os.path.join(data_dir,'emnist-byclass-test-labels-idx1-ubyte'))

    all_imgs = np.concatenate([tr_imgs, te_imgs], axis=0)
    all_lbls = np.concatenate([tr_lbls, te_lbls], axis=0)
    all_imgs = torch.from_numpy(all_imgs).unsqueeze(1).float().to(device)
    all_lbls = torch.from_numpy(all_lbls).long().to(device)

    num_classes = 62
    rng = np.random.default_rng(seed)
    extracted = []
    for cls in range(num_classes):
        idxs = np.where(all_lbls.cpu().numpy()==cls)[0]
        sel = rng.choice(idxs, size=min(200,len(idxs)), replace=False)
        extracted.extend(sel.tolist())

    base_dir = os.path.join('results','results_L2_SGD_lr0.01_5_3','KaNet5')
    model_dir = os.path.join(base_dir,'model')
    fmap_dir  = os.path.join(base_dir,'FeatureMap')
    gcam_dir  = os.path.join(base_dir,'GradCAM')

    ckpts = sorted(
        [f for f in os.listdir(model_dir) if f.endswith('.pth')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    for ckpt in ckpts:
        epoch = int(ckpt.split('_')[-1].split('.')[0])
        model = LeNet5_KAN(num_classes).to(device)
        cp = torch.load(os.path.join(model_dir,ckpt), map_location=device)
        model.load_state_dict(cp['model_state_dict'])
        model.eval()
        gradcam = GradCAM(model=model, target_layers=[model.conv2])

        for root in (fmap_dir, gcam_dir):
            for scale in ('Original','Upsampled'):
                os.makedirs(
                    os.path.join(root,f'epoch_{epoch}','extracted_random_examples',scale),
                    exist_ok=True
                )

        total = len(extracted)
        start = time.time()
        for i, idx in enumerate(extracted,1):
            img = all_imgs[idx].unsqueeze(0)
            lbl = all_lbls[idx].item()

            with torch.no_grad():
                _ = model(img)
                feat = model.last_conv_feat.squeeze(0)
                fmap_orig = feat.mean(dim=0).cpu().numpy()
                fmap_up   = F.interpolate(
                    torch.from_numpy(fmap_orig).unsqueeze(0).unsqueeze(0),
                    size=(28,28), mode='bicubic', align_corners=False
                ).squeeze().numpy()

            gcam_up_np = gradcam(input_tensor=img, targets=None)[0]
            gcam_orig = F.adaptive_avg_pool2d(
                torch.from_numpy(gcam_up_np).unsqueeze(0).unsqueeze(0),
                (10,10)
            ).squeeze().numpy()

            entries = [
                ('fmap_orig', fmap_orig, fmap_dir, 'Original'),
                ('fmap_up',   fmap_up,   fmap_dir, 'Upsampled'),
                ('gcam_orig', gcam_orig, gcam_dir, 'Original'),
                ('gcam_up',   gcam_up_np, gcam_dir, 'Upsampled')
            ]
            for name, arr, root, scale in entries:
                path = os.path.join(
                    root,
                    f'epoch_{epoch}',
                    'extracted_random_examples',
                    scale,
                    f'class_{lbl}_idx_{idx}_{name}.npy'
                )
                np.save(path, arr)

            if i % 200 == 0 or i == total:
                elapsed = time.time() - start
                avg = elapsed / i
                rem = avg * (total - i)
                print(f"Processed {i}/{total} ({i/total*100:.1f}%), "
                      f"elapsed {elapsed:.1f}s, remaining ~{rem:.1f}s")

if __name__ == '__main__':
    main()
