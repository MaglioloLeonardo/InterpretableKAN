import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import random
from pytorch_grad_cam import GradCAM

# -------------------------
# Definizione del Modello
# -------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes=62):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.last_conv_feat = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        self.last_conv_feat = x
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------
# Funzioni per leggere IDX
# -------------------------
def read_idx_images(file_path):
    with open(file_path, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return imgs

def read_idx_labels(file_path):
    with open(file_path, 'rb') as f:
        _, num = struct.unpack('>II', f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8)
    return labs

# -------------------------
# Main
# -------------------------
def main():
    seed = 14
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/home/magliolo/.cache/emnist/gzip/'

    # Caricamento dati EMNIST
    train_imgs = read_idx_images(os.path.join(data_dir, 'emnist-byclass-train-images-idx3-ubyte'))
    train_labs = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-train-labels-idx1-ubyte'))
    test_imgs  = read_idx_images(os.path.join(data_dir, 'emnist-byclass-test-images-idx3-ubyte'))
    test_labs  = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-test-labels-idx1-ubyte'))

    # Unisco train + test
    all_imgs = np.concatenate([train_imgs, test_imgs], axis=0)
    all_labs = np.concatenate([train_labs, test_labs], axis=0)
    all_imgs = torch.from_numpy(all_imgs).unsqueeze(1).float().to(device)
    all_labs = torch.from_numpy(all_labs).long().to(device)

    num_classes = 62
    # Estrai 200 esempi casuali per classe
    rng = np.random.default_rng(seed)
    extracted = []
    for cls in range(num_classes):
        idxs = np.where(all_labs.cpu().numpy() == cls)[0]
        sel = rng.choice(idxs, size=min(200, len(idxs)), replace=False)
        extracted.extend(sel.tolist())

    learning_rate = 0.01
    optimizer_type = 'SGD'
    model_variants = [
        {'norm_type': 'L2','grid_size':0,'spline_order':0},
        {'norm_type': 'None','grid_size':0,'spline_order':0}
    ]

    for variant in model_variants:
        norm_type = variant['norm_type']
        base = os.path.join(
            'results',
            f"results_{norm_type}_{optimizer_type}_lr{learning_rate}_{variant['grid_size']}_{variant['spline_order']}",
            'Standard_LeNet5'
        )
        model_dir = os.path.join(base, 'model')
        fmap_dir  = os.path.join(base, 'FeatureMap')
        gcam_dir  = os.path.join(base, 'GradCAM')

        ckpts = sorted(
            [f for f in os.listdir(model_dir) if f.endswith('.pth')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        for ckpt in ckpts:
            epoch = int(ckpt.split('_')[-1].split('.')[0])
            # Carico modello
            model = LeNet5(num_classes).to(device)
            checkpoint = torch.load(os.path.join(model_dir, ckpt), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            gradcam = GradCAM(model=model, target_layers=[model.conv2])

            # Preparo cartelle per numpy arrays
            for root in [fmap_dir, gcam_dir]:
                for scale in ['Original','Upsampled']:
                    os.makedirs(
                        os.path.join(root, f'epoch_{epoch}', 'extracted_random_examples', scale),
                        exist_ok=True
                    )

            total = len(extracted)
            start = time.time()
            for i, idx in enumerate(extracted, 1):
                img_tensor = all_imgs[idx].unsqueeze(0)
                label = all_labs[idx].item()

                # Calcolo feature maps
                with torch.no_grad():
                    _ = model(img_tensor)
                    feat = model.last_conv_feat.squeeze(0)
                    fmap_orig = feat.mean(dim=0).cpu().numpy()       # 10x10
                    fmap_up   = F.interpolate(
                        torch.from_numpy(fmap_orig).unsqueeze(0).unsqueeze(0),
                        size=(28,28), mode='bicubic', align_corners=False
                    ).squeeze().numpy()                               # 28x28

                # Calcolo GradCAM
                gcam_up  = gradcam(input_tensor=img_tensor, targets=None)[0]   # 28x28 numpy
                gcam_orig = F.adaptive_avg_pool2d(
                    torch.from_numpy(gcam_up).unsqueeze(0).unsqueeze(0), (10,10)
                ).squeeze().numpy()                                           # 10x10

                # Salvataggio esclusivamente numpy arrays
                entries = [
                    ('fmap_orig', fmap_orig, fmap_dir, 'Original'),
                    ('fmap_up',   fmap_up,   fmap_dir, 'Upsampled'),
                    ('gcam_orig', gcam_orig, gcam_dir, 'Original'),
                    ('gcam_up',   gcam_up,   gcam_dir, 'Upsampled')
                ]
                for name, arr, root, scale in entries:
                    out_dir = os.path.join(root, f'epoch_{epoch}', 'extracted_random_examples', scale)
                    npy_path = os.path.join(out_dir, f'class_{label}_idx_{idx}_{name}.npy')
                    np.save(npy_path, arr)

                # Stampa progresso
                if i % 200 == 0 or i == total:
                    elapsed = time.time() - start
                    avg = elapsed / i
                    rem = avg * (total - i)
                    print(
                        f"Processed {i}/{total} ({i/total*100:.1f}%), "
                        f"elapsed {elapsed:.1f}s, remaining ~{rem:.1f}s"
                    )

if __name__ == '__main__':
    main()
