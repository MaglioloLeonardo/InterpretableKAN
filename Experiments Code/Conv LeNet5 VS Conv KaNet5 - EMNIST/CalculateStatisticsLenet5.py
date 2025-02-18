import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import random
import csv
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Definizione del Modello
# -------------------------

class LeNet5(nn.Module):
    def __init__(self, num_classes=62):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 14x14 -> 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # 10x10 -> 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                   # Flatten
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # To store feature maps
        self.last_conv_feat = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        self.last_conv_feat = x  # Salviamo le feature maps dopo conv2 + ReLU
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------
# Definizione del Dataset in Memoria
# -------------------------

class EMNISTMemoryDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# -------------------------
# Funzioni per leggere i file IDX
# -------------------------

def read_idx_images(file_path):
    """Legge immagini in formato IDX."""
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_idx_labels(file_path):
    """Legge etichette in formato IDX."""
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# -------------------------
# Mappatura delle Classi
# -------------------------

def get_emnist_class_mapping():
    characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return {i: char for i, char in enumerate(characters)}

# -------------------------
# Funzione Principale
# -------------------------

def main():
    # Impostazione dei parametri
    learning_rate = 0.01
    optimizer_type = "SGD"
    grid_size = 0
    spline_order = 0
    norm_type = "L2"
    num_of_classes = 62
    batch_size = 128

    # Seme per riproducibilità
    seed = 12
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    mean, std = 0.1307, 0.3081

    data_dir = '/home/magliolo/.cache/emnist/gzip/'

    base_dir = os.path.join(
        'results',
        f"results_{norm_type}_{optimizer_type}_lr{learning_rate}_{grid_size}_{spline_order}",
        'Standard_LeNet5'
    )
    model_dir = os.path.join(base_dir, "model")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory del modello non trovata: {model_dir}")

    model = LeNet5(num_classes=num_of_classes).to(device)

    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not checkpoints:
        raise FileNotFoundError(f"Nessun checkpoint trovato nella directory: {model_dir}")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint caricato: {checkpoint_path}, Epoch: {checkpoint['epoch']}")
    model.eval()

    # Leggi i dati di test
    test_images_path = os.path.join(data_dir, 'emnist-byclass-test-images-idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 'emnist-byclass-test-labels-idx1-ubyte')
    print("Leggendo i dati di test...")
    images_test = read_idx_images(test_images_path)
    labels_test = read_idx_labels(test_labels_path)

    # Leggi i dati di train (analoghi a test, assicurarsi di avere i file)
    train_images_path = os.path.join(data_dir, 'emnist-byclass-train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'emnist-byclass-train-labels-idx1-ubyte')
    print("Leggendo i dati di train...")
    images_train = read_idx_images(train_images_path)
    labels_train = read_idx_labels(train_labels_path)

    # Conversione in tensori (senza normalizzazione tra 0 e 1, come richiesto)
    # Non normalizziamo per [0,1], lasceremo i valori come uint8 convertiti in float
    # Anche se il modello è stato addestrato su dati normalizzati. Questo potrebbe alterare i risultati,
    # ma la richiesta era di non normalizzare tra 0 e 1 per l'estrazione feature.
    # Il modello funzionerà comunque (darà predizioni sballate, ma l'utente ha richiesto così).
    train_images_tensor = torch.from_numpy(images_train.copy()).unsqueeze(1).float().to(device)
    train_labels_tensor = torch.from_numpy(labels_train.copy()).long().to(device)
    test_images_tensor = torch.from_numpy(images_test.copy()).unsqueeze(1).float().to(device)
    test_labels_tensor = torch.from_numpy(labels_test.copy()).long().to(device)

    train_dataset = EMNISTMemoryDataset(train_images_tensor, train_labels_tensor)
    test_dataset = EMNISTMemoryDataset(test_images_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_mapping = get_emnist_class_mapping()

    # Accumulatori per ciascuna categoria:
    # train_correct, train_incorrect, test_correct, test_incorrect
    accumulators = {
        'train_correct': {'sum_means': torch.tensor(0.0, device=device), 
                          'sum_vars': torch.tensor(0.0, device=device), 
                          'sum_maxs': torch.tensor(0.0, device=device), 
                          'count': torch.tensor(0, device=device)},
        'train_incorrect': {'sum_means': torch.tensor(0.0, device=device), 
                            'sum_vars': torch.tensor(0.0, device=device), 
                            'sum_maxs': torch.tensor(0.0, device=device), 
                            'count': torch.tensor(0, device=device)},
        'test_correct': {'sum_means': torch.tensor(0.0, device=device), 
                         'sum_vars': torch.tensor(0.0, device=device), 
                         'sum_maxs': torch.tensor(0.0, device=device), 
                         'count': torch.tensor(0, device=device)},
        'test_incorrect': {'sum_means': torch.tensor(0.0, device=device), 
                           'sum_vars': torch.tensor(0.0, device=device), 
                           'sum_maxs': torch.tensor(0.0, device=device), 
                           'count': torch.tensor(0, device=device)},
    }

    # Funzione per processare un loader (train o test)
    def process_loader(loader, is_test=False):
        category = 'test' if is_test else 'train'
        for data, target in loader:
            with torch.no_grad():
                output = model(data)
                pred = output.argmax(dim=1)
                correct_mask = (pred == target)

                # Estrazione feature maps dall'ultimo layer conv2 dopo ReLU
                # feature maps: [B, 16, 10, 10] prima del pool2
                feats = model.last_conv_feat  # Abbiamo salvato in forward

                # Calcoliamo mean, var, max per ogni immagine
                # feats shape: (batch_size, 16, H, W)
                # Flatteniamo su tutti i canali e pixel
                B = feats.size(0)
                val = feats.view(B, -1)  # (B, 16*10*10)
                means = torch.mean(val, dim=1)  # [B]
                vars_ = torch.var(val, dim=1)   # [B]
                maxs = torch.max(val, dim=1)[0] # [B]

                # Separare corretti e errati
                correct_indices = correct_mask.nonzero(as_tuple=False).squeeze(1)
                incorrect_indices = (~correct_mask).nonzero(as_tuple=False).squeeze(1)

                # Aggiorna gli accumulatori usando operazioni vettoriali
                if correct_indices.numel() > 0:
                    accumulators[f'{category}_correct']['sum_means'] += means[correct_indices].sum()
                    accumulators[f'{category}_correct']['sum_vars'] += vars_[correct_indices].sum()
                    accumulators[f'{category}_correct']['sum_maxs'] += maxs[correct_indices].sum()
                    accumulators[f'{category}_correct']['count'] += correct_indices.numel()

                if incorrect_indices.numel() > 0:
                    accumulators[f'{category}_incorrect']['sum_means'] += means[incorrect_indices].sum()
                    accumulators[f'{category}_incorrect']['sum_vars'] += vars_[incorrect_indices].sum()
                    accumulators[f'{category}_incorrect']['sum_maxs'] += maxs[incorrect_indices].sum()
                    accumulators[f'{category}_incorrect']['count'] += incorrect_indices.numel()

    # Processa train e test
    print("Processo il training set...")
    process_loader(train_loader, is_test=False)
    print("Processo il test set...")
    process_loader(test_loader, is_test=True)

    # Funzione per calcolare le statistiche aggregate e salvare in CSV
    def save_stats_to_csv(filename, data):
        if data['count'].item() == 0:
            mean_of_means = 0.0
            mean_of_vars = 0.0
            mean_of_max = 0.0
        else:
            mean_of_means = (data['sum_means'] / data['count']).item()
            mean_of_vars = (data['sum_vars'] / data['count']).item()
            mean_of_max = (data['sum_maxs'] / data['count']).item()

        # Salva in CSV con una sola riga: mean_of_means, mean_of_vars, mean_of_max
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["mean_intensity", "variance_intensity", "max_intensity"])
            writer.writerow([mean_of_means, mean_of_vars, mean_of_max])

        print(f"Salvato: {filename}")
        # Print contenuto
        print("Contenuto CSV:")
        print("mean_intensity,variance_intensity,max_intensity")
        print(f"{mean_of_means},{mean_of_vars},{mean_of_max}")

    # Salva i quattro CSV
    save_stats_to_csv("train_correct.csv", accumulators['train_correct'])
    save_stats_to_csv("train_incorrect.csv", accumulators['train_incorrect'])
    save_stats_to_csv("test_correct.csv", accumulators['test_correct'])
    save_stats_to_csv("test_incorrect.csv", accumulators['test_incorrect'])

if __name__ == "__main__":
    main()