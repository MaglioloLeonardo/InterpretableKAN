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
# Definizione del Modello KAN
# -------------------------

# Importa i moduli personalizzati per KAN
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

        # Fully Connected Layers
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  

        # Variabile per salvare le feature maps dopo conv2 + ReLU
        self.last_conv_feat = None

    def forward(self, x):
        # Passo 1: conv + pooling
        x = self.conv1(x)
        x = self.pool1(x)
        
        # Passo 2: conv + pooling
        x = self.conv2(x)
        self.last_conv_feat = x  # Salva le feature maps dopo conv2 + ReLU
        x = self.pool2(x)
        
        # Flatten
        x = x.contiguous().reshape(x.size(0), -1)

        # Fully Connected Layers con ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output Layer (senza attivazione)
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
    """
    Mappatura delle classi EMNIST ai caratteri corrispondenti.
    EMNIST ByClass ha 62 classi: 0-9, 10-35 A-Z, 36-61 a-z
    """
    characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return {i: char for i, char in enumerate(characters)}

# -------------------------
# Funzione Principale
# -------------------------

def main():
    # Impostazione dei parametri
    learning_rate = 0.01
    optimizer_type = "SGD"
    grid_size = 5
    spline_order = 3
    norm_type = "L2"
    num_of_classes = 62
    batch_size = 128  # Aumenta il batch_size per sfruttare meglio la GPU

    # Seme per riproducibilità
    seed = 12
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    mean, std = 0.1307, 0.3081  # Valori di normalizzazione standard per EMNIST

    data_dir = '/home/magliolo/.cache/emnist/gzip/'

    base_dir = os.path.join(
        'results',
        f"results_{norm_type}_{optimizer_type}_lr{learning_rate}_{grid_size}_{spline_order}",
        'KaNet5'
    )
    model_dir = os.path.join(base_dir, "model")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory del modello non trovata: {model_dir}")

    # Inizializza il modello KAN
    model = LeNet5_KAN(num_classes=num_of_classes).to(device)

    # Carica il checkpoint più recente
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not checkpoints:
        raise FileNotFoundError(f"Nessun checkpoint trovato nella directory: {model_dir}")
    # Estrai il numero dell'epoch dal nome del file, assumendo formato 'model_epoch_{epoch}.pth'
    try:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except ValueError:
        # Se il formato del checkpoint è diverso, usa semplicemente il file più recente
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint caricato: {checkpoint_path}, Epoch: {checkpoint.get('epoch', 'Unknown')}")
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
    # Nota: Il modello è stato addestrato su dati normalizzati, ma l'utente richiede di non normalizzare
    # Pertanto, le statistiche potrebbero essere influenzate negativamente
    train_images_tensor = torch.from_numpy(images_train.copy()).unsqueeze(1).float()
    train_labels_tensor = torch.from_numpy(labels_train.copy()).long()
    test_images_tensor = torch.from_numpy(images_test.copy()).unsqueeze(1).float()
    test_labels_tensor = torch.from_numpy(labels_test.copy()).long()

    # Crea il dataset in memoria
    train_dataset = EMNISTMemoryDataset(train_images_tensor, train_labels_tensor)
    test_dataset = EMNISTMemoryDataset(test_images_tensor, test_labels_tensor)

    # Crea i DataLoader con num_workers=0 per evitare problemi di CUDA nei worker
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

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
                # Sposta i dati e le etichette sulla GPU
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(data)
                pred = output.argmax(dim=1)
                correct_mask = (pred == target)

                # Estrazione feature maps dall'ultimo layer conv2 dopo ReLU
                feats = model.last_conv_feat  # Abbiamo salvato in forward

                # Calcola mean, var, max per ogni immagine
                # feats shape: (batch_size, 16, 8, 8)
                # Flatteniamo su tutti i canali e pixel
                B = feats.size(0)
                val = feats.reshape(B, -1)  # (B, 16*8*8)
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

    # Crea la directory base_dir se non esiste
    os.makedirs(base_dir, exist_ok=True)

    # Salva i quattro CSV nella directory base_dir
    save_stats_to_csv(os.path.join(base_dir, "train_correct.csv"), accumulators['train_correct'])
    save_stats_to_csv(os.path.join(base_dir, "train_incorrect.csv"), accumulators['train_incorrect'])
    save_stats_to_csv(os.path.join(base_dir, "test_correct.csv"), accumulators['test_correct'])
    save_stats_to_csv(os.path.join(base_dir, "test_incorrect.csv"), accumulators['test_incorrect'])

    print("Tutti i file CSV sono stati salvati correttamente.")

if __name__ == "__main__":
    main()