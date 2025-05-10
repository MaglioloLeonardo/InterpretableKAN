import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import random
import csv
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Importa il modulo GradCAM dalla libreria pytorch-gradcam
from pytorch_grad_cam import GradCAM

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
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        # Salva le feature maps per eventuali elaborazioni (es. per calcolare le statistiche)
        self.last_conv_feat = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        self.last_conv_feat = x  # Dimensione attesa: [B, 16, 10, 10]
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
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
# Mappatura delle Classi EMNIST
# -------------------------
def get_emnist_class_mapping():
    characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return {i: char for i, char in enumerate(characters)}

# -------------------------
# Funzioni per salvare le statistiche in CSV (globali e per classe)
# -------------------------
def save_stats_to_csv(full_path, data):
    if data['count'].item() == 0:
        mean_of_means = 0.0
        mean_of_vars = 0.0
        mean_of_max = 0.0
    else:
        mean_of_means = (data['sum_means'] / data['count']).item()
        mean_of_vars = (data['sum_vars'] / data['count']).item()
        mean_of_max = (data['sum_maxs'] / data['count']).item()
    with open(full_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["mean_intensity", "variance_intensity", "max_intensity"])
        writer.writerow([mean_of_means, mean_of_vars, mean_of_max])
    print(f"Salvato CSV: {full_path}")

def save_stats_to_csv_by_class(full_path, data_by_class, class_mapping):
    with open(full_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_char", "mean_intensity", "variance_intensity", "max_intensity", "count"])
        for cls in sorted(data_by_class.keys()):
            stats = data_by_class[cls]
            count = stats['count']
            if count == 0:
                mean_of_means = 0.0
                mean_of_vars = 0.0
                mean_of_max = 0.0
            else:
                mean_of_means = stats['sum_means'] / count
                mean_of_vars = stats['sum_vars'] / count
                mean_of_max = stats['sum_maxs'] / count
            writer.writerow([cls, class_mapping.get(cls, "NA"), mean_of_means, mean_of_vars, mean_of_max, count])
    print(f"Salvato CSV: {full_path}")

# -------------------------
# Funzione Principale
# -------------------------
def main():
    # Parametri
    learning_rate = 0.01
    optimizer_type = "SGD"
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

    # NOTA: Per il calcolo delle heatmap (sia feature map che GradCAM) non applichiamo normalizzazione
    data_dir = '/home/magliolo/.cache/emnist/gzip/'

    # Caricamento dei dati EMNIST
    print("Lettura dati test...")
    test_images = read_idx_images(os.path.join(data_dir, 'emnist-byclass-test-images-idx3-ubyte'))
    test_labels = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-test-labels-idx1-ubyte'))
    print("Lettura dati train...")
    train_images = read_idx_images(os.path.join(data_dir, 'emnist-byclass-train-images-idx3-ubyte'))
    train_labels = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-train-labels-idx1-ubyte'))

    # Conversione in tensori (senza normalizzazione)
    train_images_tensor = torch.from_numpy(train_images.copy()).unsqueeze(1).float().to(device)
    train_labels_tensor = torch.from_numpy(train_labels.copy()).long().to(device)
    test_images_tensor = torch.from_numpy(test_images.copy()).unsqueeze(1).float().to(device)
    test_labels_tensor = torch.from_numpy(test_labels.copy()).long().to(device)

    train_dataset = EMNISTMemoryDataset(train_images_tensor, train_labels_tensor)
    test_dataset = EMNISTMemoryDataset(test_images_tensor, test_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_mapping = get_emnist_class_mapping()

    # Definizione delle varianti del modello
    model_variants = [
        {"norm_type": "L2", "grid_size": 0, "spline_order": 0},
        {"norm_type": "None", "grid_size": 0, "spline_order": 0}
    ]

    for variant in model_variants:
        norm_type = variant["norm_type"]
        grid_size = variant["grid_size"]
        spline_order = variant["spline_order"]

        print(f"\n===== Elaborazione variante: norm_type='{norm_type}', grid_size={grid_size}, spline_order={spline_order} =====")

        base_dir = os.path.join(
            'results',
            f"results_{norm_type}_{optimizer_type}_lr{learning_rate}_{grid_size}_{spline_order}",
            'Standard_LeNet5'
        )
        model_dir = os.path.join(base_dir, "model")
        # Due directory distinte: una per i risultati delle GradCAM e una per le feature map
        gradcam_dir = os.path.join(base_dir, "GradCAM")
        featuremap_dir = os.path.join(base_dir, "FeatureMap")
        os.makedirs(gradcam_dir, exist_ok=True)
        os.makedirs(featuremap_dir, exist_ok=True)

        # Lista dei checkpoint (ad es. "checkpoint_epoch_50.pth")
        checkpoint_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')],
                                  key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Crea un modello (lo stesso per tutti i checkpoint)
        model = LeNet5(num_classes=num_of_classes).to(device)

        for ckpt_file in checkpoint_files:
            epoch_num = int(ckpt_file.split('_')[-1].split('.')[0])
            model_type = "Standard_LeNet5"

            # ============================
            # Creazione delle cartelle di salvataggio
            # ============================

            # Per FeatureMap
            epoch_folder_featuremap = os.path.join(featuremap_dir, f"epoch_{epoch_num}")
            orig_folder_featuremap = os.path.join(epoch_folder_featuremap, "Original")
            orig_mean_folder_featuremap = os.path.join(orig_folder_featuremap, "Mean")
            orig_variance_folder_featuremap = os.path.join(orig_folder_featuremap, "Variance")
            up_folder_featuremap = os.path.join(epoch_folder_featuremap, "Upsampled")
            up_mean_folder_featuremap = os.path.join(up_folder_featuremap, "Mean")
            up_variance_folder_featuremap = os.path.join(up_folder_featuremap, "Variance")
            os.makedirs(orig_mean_folder_featuremap, exist_ok=True)
            os.makedirs(orig_variance_folder_featuremap, exist_ok=True)
            os.makedirs(up_mean_folder_featuremap, exist_ok=True)
            os.makedirs(up_variance_folder_featuremap, exist_ok=True)

            # Per GradCAM
            epoch_folder_gradcam = os.path.join(gradcam_dir, f"epoch_{epoch_num}")
            orig_folder_gradcam = os.path.join(epoch_folder_gradcam, "Original")
            orig_mean_folder_gradcam = os.path.join(orig_folder_gradcam, "Mean")
            orig_variance_folder_gradcam = os.path.join(orig_folder_gradcam, "Variance")
            up_folder_gradcam = os.path.join(epoch_folder_gradcam, "Upsampled")
            up_mean_folder_gradcam = os.path.join(up_folder_gradcam, "Mean")
            up_variance_folder_gradcam = os.path.join(up_folder_gradcam, "Variance")
            os.makedirs(orig_mean_folder_gradcam, exist_ok=True)
            os.makedirs(orig_variance_folder_gradcam, exist_ok=True)
            os.makedirs(up_mean_folder_gradcam, exist_ok=True)
            os.makedirs(up_variance_folder_gradcam, exist_ok=True)

            # Carica il checkpoint
            checkpoint_path = os.path.join(model_dir, ckpt_file)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Checkpoint caricato: {checkpoint_path}, Epoch: {checkpoint.get('epoch', 'Unknown')}")
            model.eval()

            # ============================
            # Inizializzazione degli accumulatori
            # ============================
            # Per FeatureMap – dimensione originale (10x10)
            heatmap_orig_accum_by_class_train = {cls: torch.zeros((10, 10), device=device) for cls in range(num_of_classes)}
            heatmap_orig_squared_accum_by_class_train = {cls: torch.zeros((10, 10), device=device) for cls in range(num_of_classes)}
            heatmap_orig_count_by_class_train = {cls: 0 for cls in range(num_of_classes)}
            heatmap_orig_accum_by_class_test = {cls: torch.zeros((10, 10), device=device) for cls in range(num_of_classes)}
            heatmap_orig_squared_accum_by_class_test = {cls: torch.zeros((10, 10), device=device) for cls in range(num_of_classes)}
            heatmap_orig_count_by_class_test = {cls: 0 for cls in range(num_of_classes)}

            # Per FeatureMap – upsampled (28x28)
            heatmap_up_accum_by_class_train = {cls: torch.zeros((28, 28), device=device) for cls in range(num_of_classes)}
            heatmap_up_squared_accum_by_class_train = {cls: torch.zeros((28, 28), device=device) for cls in range(num_of_classes)}
            heatmap_up_count_by_class_train = {cls: 0 for cls in range(num_of_classes)}
            heatmap_up_accum_by_class_test = {cls: torch.zeros((28, 28), device=device) for cls in range(num_of_classes)}
            heatmap_up_squared_accum_by_class_test = {cls: torch.zeros((28, 28), device=device) for cls in range(num_of_classes)}
            heatmap_up_count_by_class_test = {cls: 0 for cls in range(num_of_classes)}

            # Per GradCAM – upsampled (28x28) [quello restituito dalla libreria]
            gradcam_up_accum_by_class_train = {cls: torch.zeros((28, 28), device=device) for cls in range(num_of_classes)}
            gradcam_up_squared_accum_by_class_train = {cls: torch.zeros((28, 28), device=device) for cls in range(num_of_classes)}
            gradcam_up_count_by_class_train = {cls: 0 for cls in range(num_of_classes)}
            gradcam_up_accum_by_class_test = {cls: torch.zeros((28, 28), device=device) for cls in range(num_of_classes)}
            gradcam_up_squared_accum_by_class_test = {cls: torch.zeros((28, 28), device=device) for cls in range(num_of_classes)}
            gradcam_up_count_by_class_test = {cls: 0 for cls in range(num_of_classes)}

            # Per GradCAM – originale (10x10, ottenute downscalando il risultato upsampled)
            gradcam_orig_accum_by_class_train = {cls: torch.zeros((10, 10), device=device) for cls in range(num_of_classes)}
            gradcam_orig_squared_accum_by_class_train = {cls: torch.zeros((10, 10), device=device) for cls in range(num_of_classes)}
            gradcam_orig_count_by_class_train = {cls: 0 for cls in range(num_of_classes)}
            gradcam_orig_accum_by_class_test = {cls: torch.zeros((10, 10), device=device) for cls in range(num_of_classes)}
            gradcam_orig_squared_accum_by_class_test = {cls: torch.zeros((10, 10), device=device) for cls in range(num_of_classes)}
            gradcam_orig_count_by_class_test = {cls: 0 for cls in range(num_of_classes)}

            # Crea l'oggetto GradCAM per il modello corrente (nota: rimosso il parametro use_cuda)
            gradcam_obj = GradCAM(model=model, target_layers=[model.conv2])

            # ============================
            # Funzione per processare un loader (train o test)
            # ============================
            def process_loader(loader, is_test=False):
                for data, target in loader:
                    # Calcola l'output e le feature map
                    with torch.no_grad():
                        output = model(data)
                        # Ottieni le feature map originali dal layer conv2 (già salvate in model.last_conv_feat)
                        feats = model.last_conv_feat  # [B, 16, 10, 10]
                        B = feats.size(0)
                        # Calcola la heatmap: media sui canali
                        heatmaps_orig = torch.mean(feats, dim=1)  # [B, 10, 10]
                        # Upsample delle heatmap usando interpolazione bicubica (per evitare artefatti)
                        heatmaps_up = F.interpolate(heatmaps_orig.unsqueeze(1), size=(28, 28), mode='bicubic', align_corners=False).squeeze(1)
                    
                    # Accumula per ciascuna immagine (FeatureMap)
                    for i in range(B):
                        cls = target[i].item()
                        if is_test:
                            heatmap_orig_accum_by_class_test[cls] += heatmaps_orig[i]
                            heatmap_orig_squared_accum_by_class_test[cls] += heatmaps_orig[i] ** 2
                            heatmap_orig_count_by_class_test[cls] += 1

                            heatmap_up_accum_by_class_test[cls] += heatmaps_up[i]
                            heatmap_up_squared_accum_by_class_test[cls] += heatmaps_up[i] ** 2
                            heatmap_up_count_by_class_test[cls] += 1
                        else:
                            heatmap_orig_accum_by_class_train[cls] += heatmaps_orig[i]
                            heatmap_orig_squared_accum_by_class_train[cls] += heatmaps_orig[i] ** 2
                            heatmap_orig_count_by_class_train[cls] += 1

                            heatmap_up_accum_by_class_train[cls] += heatmaps_up[i]
                            heatmap_up_squared_accum_by_class_train[cls] += heatmaps_up[i] ** 2
                            heatmap_up_count_by_class_train[cls] += 1

                    # Calcola le mappe GradCAM per il batch
                    # NOTA: il risultato restituito dalla libreria è già upsampled (28x28)
                    gradcam_maps_up_np = gradcam_obj(input_tensor=data, targets=None)
                    gradcam_maps_up = torch.from_numpy(gradcam_maps_up_np).to(device).float()  # [B, 28, 28]
                    # Ottieni la versione originale (10x10) downscalando (adaptive_avg_pool2d)
                    gradcam_maps_orig = F.adaptive_avg_pool2d(gradcam_maps_up.unsqueeze(1), output_size=(10, 10)).squeeze(1)
                    
                    for i in range(B):
                        cls = target[i].item()
                        if is_test:
                            gradcam_up_accum_by_class_test[cls] += gradcam_maps_up[i]
                            gradcam_up_squared_accum_by_class_test[cls] += gradcam_maps_up[i] ** 2
                            gradcam_up_count_by_class_test[cls] += 1

                            gradcam_orig_accum_by_class_test[cls] += gradcam_maps_orig[i]
                            gradcam_orig_squared_accum_by_class_test[cls] += gradcam_maps_orig[i] ** 2
                            gradcam_orig_count_by_class_test[cls] += 1
                        else:
                            gradcam_up_accum_by_class_train[cls] += gradcam_maps_up[i]
                            gradcam_up_squared_accum_by_class_train[cls] += gradcam_maps_up[i] ** 2
                            gradcam_up_count_by_class_train[cls] += 1

                            gradcam_orig_accum_by_class_train[cls] += gradcam_maps_orig[i]
                            gradcam_orig_squared_accum_by_class_train[cls] += gradcam_maps_orig[i] ** 2
                            gradcam_orig_count_by_class_train[cls] += 1

            print("Processo il training set...")
            process_loader(train_loader, is_test=False)
            print("Processo il test set...")
            process_loader(test_loader, is_test=True)

            # ============================
            # Salvataggio dei risultati per FeatureMap
            # ============================
            for cls in range(num_of_classes):
                # --- FeatureMap: Versione ORIGINALE (10x10) ---
                if heatmap_orig_count_by_class_train[cls] > 0:
                    mean_heatmap_orig_train = heatmap_orig_accum_by_class_train[cls] / heatmap_orig_count_by_class_train[cls]
                    var_heatmap_orig_train = (heatmap_orig_squared_accum_by_class_train[cls] / heatmap_orig_count_by_class_train[cls]) - (mean_heatmap_orig_train ** 2)
                    # Salvataggio per il training set
                    img_filename = os.path.join(orig_mean_folder_featuremap, f"{model_type}_train_mean_featuremap_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(mean_heatmap_orig_train.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(orig_mean_folder_featuremap, f"{model_type}_train_mean_featuremap_class_{cls}.csv")
                    np.savetxt(csv_filename, mean_heatmap_orig_train.cpu().numpy(), delimiter=",")
                    print(f"Salvata mean featuremap (Original, train) per classe {cls}: {img_filename}")

                    img_filename = os.path.join(orig_variance_folder_featuremap, f"{model_type}_train_variance_featuremap_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(var_heatmap_orig_train.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(orig_variance_folder_featuremap, f"{model_type}_train_variance_featuremap_class_{cls}.csv")
                    np.savetxt(csv_filename, var_heatmap_orig_train.cpu().numpy(), delimiter=",")
                    print(f"Salvata variance featuremap (Original, train) per classe {cls}: {img_filename}")
                if heatmap_orig_count_by_class_test[cls] > 0:
                    mean_heatmap_orig_test = heatmap_orig_accum_by_class_test[cls] / heatmap_orig_count_by_class_test[cls]
                    var_heatmap_orig_test = (heatmap_orig_squared_accum_by_class_test[cls] / heatmap_orig_count_by_class_test[cls]) - (mean_heatmap_orig_test ** 2)
                    img_filename = os.path.join(orig_mean_folder_featuremap, f"{model_type}_test_mean_featuremap_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(mean_heatmap_orig_test.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(orig_mean_folder_featuremap, f"{model_type}_test_mean_featuremap_class_{cls}.csv")
                    np.savetxt(csv_filename, mean_heatmap_orig_test.cpu().numpy(), delimiter=",")
                    print(f"Salvata mean featuremap (Original, test) per classe {cls}: {img_filename}")
                    img_filename = os.path.join(orig_variance_folder_featuremap, f"{model_type}_test_variance_featuremap_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(var_heatmap_orig_test.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(orig_variance_folder_featuremap, f"{model_type}_test_variance_featuremap_class_{cls}.csv")
                    np.savetxt(csv_filename, var_heatmap_orig_test.cpu().numpy(), delimiter=",")
                    print(f"Salvata variance featuremap (Original, test) per classe {cls}: {img_filename}")

                # --- FeatureMap: Versione UPSAMPLED (28x28) ---
                if heatmap_up_count_by_class_train[cls] > 0:
                    mean_heatmap_up_train = heatmap_up_accum_by_class_train[cls] / heatmap_up_count_by_class_train[cls]
                    var_heatmap_up_train = (heatmap_up_squared_accum_by_class_train[cls] / heatmap_up_count_by_class_train[cls]) - (mean_heatmap_up_train ** 2)
                    img_filename = os.path.join(up_mean_folder_featuremap, f"{model_type}_train_mean_featuremap_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(mean_heatmap_up_train.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(up_mean_folder_featuremap, f"{model_type}_train_mean_featuremap_class_{cls}.csv")
                    np.savetxt(csv_filename, mean_heatmap_up_train.cpu().numpy(), delimiter=",")
                    print(f"Salvata mean featuremap (Upsampled, train) per classe {cls}: {img_filename}")

                    img_filename = os.path.join(up_variance_folder_featuremap, f"{model_type}_train_variance_featuremap_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(var_heatmap_up_train.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(up_variance_folder_featuremap, f"{model_type}_train_variance_featuremap_class_{cls}.csv")
                    np.savetxt(csv_filename, var_heatmap_up_train.cpu().numpy(), delimiter=",")
                    print(f"Salvata variance featuremap (Upsampled, train) per classe {cls}: {img_filename}")
                if heatmap_up_count_by_class_test[cls] > 0:
                    mean_heatmap_up_test = heatmap_up_accum_by_class_test[cls] / heatmap_up_count_by_class_test[cls]
                    var_heatmap_up_test = (heatmap_up_squared_accum_by_class_test[cls] / heatmap_up_count_by_class_test[cls]) - (mean_heatmap_up_test ** 2)
                    img_filename = os.path.join(up_mean_folder_featuremap, f"{model_type}_test_mean_featuremap_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(mean_heatmap_up_test.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(up_mean_folder_featuremap, f"{model_type}_test_mean_featuremap_class_{cls}.csv")
                    np.savetxt(csv_filename, mean_heatmap_up_test.cpu().numpy(), delimiter=",")
                    print(f"Salvata mean featuremap (Upsampled, test) per classe {cls}: {img_filename}")
                    img_filename = os.path.join(up_variance_folder_featuremap, f"{model_type}_test_variance_featuremap_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(var_heatmap_up_test.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(up_variance_folder_featuremap, f"{model_type}_test_variance_featuremap_class_{cls}.csv")
                    np.savetxt(csv_filename, var_heatmap_up_test.cpu().numpy(), delimiter=",")
                    print(f"Salvato variance featuremap (Upsampled, test) per classe {cls}: {img_filename}")

            # ============================
            # Salvataggio dei risultati per GradCAM
            # ============================
            for cls in range(num_of_classes):
                # --- GradCAM: Versione UPSAMPLED (28x28) ---
                if gradcam_up_count_by_class_train[cls] > 0:
                    mean_gradcam_up_train = gradcam_up_accum_by_class_train[cls] / gradcam_up_count_by_class_train[cls]
                    var_gradcam_up_train = (gradcam_up_squared_accum_by_class_train[cls] / gradcam_up_count_by_class_train[cls]) - (mean_gradcam_up_train ** 2)
                    img_filename = os.path.join(up_mean_folder_gradcam, f"{model_type}_train_mean_gradcam_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(mean_gradcam_up_train.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(up_mean_folder_gradcam, f"{model_type}_train_mean_gradcam_class_{cls}.csv")
                    np.savetxt(csv_filename, mean_gradcam_up_train.cpu().numpy(), delimiter=",")
                    print(f"Salvata mean GradCAM (Upsampled, train) per classe {cls}: {img_filename}")

                    img_filename = os.path.join(up_variance_folder_gradcam, f"{model_type}_train_variance_gradcam_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(var_gradcam_up_train.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(up_variance_folder_gradcam, f"{model_type}_train_variance_gradcam_class_{cls}.csv")
                    np.savetxt(csv_filename, var_gradcam_up_train.cpu().numpy(), delimiter=",")
                    print(f"Salvata variance GradCAM (Upsampled, train) per classe {cls}: {img_filename}")
                if gradcam_up_count_by_class_test[cls] > 0:
                    mean_gradcam_up_test = gradcam_up_accum_by_class_test[cls] / gradcam_up_count_by_class_test[cls]
                    var_gradcam_up_test = (gradcam_up_squared_accum_by_class_test[cls] / gradcam_up_count_by_class_test[cls]) - (mean_gradcam_up_test ** 2)
                    img_filename = os.path.join(up_mean_folder_gradcam, f"{model_type}_test_mean_gradcam_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(mean_gradcam_up_test.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(up_mean_folder_gradcam, f"{model_type}_test_mean_gradcam_class_{cls}.csv")
                    np.savetxt(csv_filename, mean_gradcam_up_test.cpu().numpy(), delimiter=",")
                    print(f"Salvata mean GradCAM (Upsampled, test) per classe {cls}: {img_filename}")
                    img_filename = os.path.join(up_variance_folder_gradcam, f"{model_type}_test_variance_gradcam_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(var_gradcam_up_test.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(up_variance_folder_gradcam, f"{model_type}_test_variance_gradcam_class_{cls}.csv")
                    np.savetxt(csv_filename, var_gradcam_up_test.cpu().numpy(), delimiter=",")
                    print(f"Salvato variance GradCAM (Upsampled, test) per classe {cls}: {img_filename}")

                # --- GradCAM: Versione ORIGINALE (10x10) ---
                if gradcam_orig_count_by_class_train[cls] > 0:
                    mean_gradcam_orig_train = gradcam_orig_accum_by_class_train[cls] / gradcam_orig_count_by_class_train[cls]
                    var_gradcam_orig_train = (gradcam_orig_squared_accum_by_class_train[cls] / gradcam_orig_count_by_class_train[cls]) - (mean_gradcam_orig_train ** 2)
                    img_filename = os.path.join(orig_mean_folder_gradcam, f"{model_type}_train_mean_gradcam_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(mean_gradcam_orig_train.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(orig_mean_folder_gradcam, f"{model_type}_train_mean_gradcam_class_{cls}.csv")
                    np.savetxt(csv_filename, mean_gradcam_orig_train.cpu().numpy(), delimiter=",")
                    print(f"Salvata mean GradCAM (Original, train) per classe {cls}: {img_filename}")

                    img_filename = os.path.join(orig_variance_folder_gradcam, f"{model_type}_train_variance_gradcam_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(var_gradcam_orig_train.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(orig_variance_folder_gradcam, f"{model_type}_train_variance_gradcam_class_{cls}.csv")
                    np.savetxt(csv_filename, var_gradcam_orig_train.cpu().numpy(), delimiter=",")
                    print(f"Salvata variance GradCAM (Original, train) per classe {cls}: {img_filename}")
                if gradcam_orig_count_by_class_test[cls] > 0:
                    mean_gradcam_orig_test = gradcam_orig_accum_by_class_test[cls] / gradcam_orig_count_by_class_test[cls]
                    var_gradcam_orig_test = (gradcam_orig_squared_accum_by_class_test[cls] / gradcam_orig_count_by_class_test[cls]) - (mean_gradcam_orig_test ** 2)
                    img_filename = os.path.join(orig_mean_folder_gradcam, f"{model_type}_test_mean_gradcam_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(mean_gradcam_orig_test.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(orig_mean_folder_gradcam, f"{model_type}_test_mean_gradcam_class_{cls}.csv")
                    np.savetxt(csv_filename, mean_gradcam_orig_test.cpu().numpy(), delimiter=",")
                    print(f"Salvata mean GradCAM (Original, test) per classe {cls}: {img_filename}")
                    img_filename = os.path.join(orig_variance_folder_gradcam, f"{model_type}_test_variance_gradcam_class_{cls}.png")
                    plt.figure(frameon=False)
                    plt.imshow(var_gradcam_orig_test.cpu().numpy(), cmap='jet')
                    plt.axis('off')
                    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    csv_filename = os.path.join(orig_variance_folder_gradcam, f"{model_type}_test_variance_gradcam_class_{cls}.csv")
                    np.savetxt(csv_filename, var_gradcam_orig_test.cpu().numpy(), delimiter=",")
                    print(f"Salvato variance GradCAM (Original, test) per classe {cls}: {img_filename}")

        # Fine ciclo sui checkpoint
    # Fine ciclo sulle varianti

if __name__ == "__main__":
    main()
