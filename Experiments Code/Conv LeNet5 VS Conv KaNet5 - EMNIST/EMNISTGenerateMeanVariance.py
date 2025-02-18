import os
import numpy as np
import struct
import random
import csv
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------
# Funzioni per leggere i file IDX
# -------------------------
def read_idx_images(file_path):
    """Legge le immagini in formato IDX."""
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_idx_labels(file_path):
    """Legge le etichette in formato IDX."""
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# -------------------------
# Funzione di elaborazione per calcolare media e varianza per classe
# -------------------------
def process_dataset(images, labels, split_name, output_root):
    """
    Calcola l'immagine media e la varianza per ciascuna classe utilizzando tutte le immagini
    (senza suddividere in train e test) e applica la trasformazione:
      - Flip orizzontale
      - Rotazione di 90° in senso antiorario
    Salva le immagini trasformate e i dati CSV.
    
    Parametri:
      - images: array numpy di shape [N, H, W]
      - labels: array numpy di shape [N]
      - split_name: 'Total' (per indicare che vengono usati tutti i dati)
      - output_root: directory di output, es. "results/E-MNIST-PROCESSING"
    """
    num_classes = 62  # EMNIST ByClass ha 62 classi
    H, W = images.shape[1], images.shape[2]
    
    # Inizializza gli accumulatori per la somma, la somma dei quadrati e il conteggio per ciascuna classe
    sum_images = [np.zeros((H, W), dtype=np.float64) for _ in range(num_classes)]
    sum_sq_images = [np.zeros((H, W), dtype=np.float64) for _ in range(num_classes)]
    count = [0 for _ in range(num_classes)]
    
    N = images.shape[0]
    for i in range(N):
        cls = int(labels[i])
        img = images[i].astype(np.float64)
        sum_images[cls] += img
        sum_sq_images[cls] += img ** 2
        count[cls] += 1

    # Per ogni classe calcola la media e la varianza
    for cls in range(num_classes):
        if count[cls] > 0:
            mean_img = sum_images[cls] / count[cls]
            var_img = (sum_sq_images[cls] / count[cls]) - (mean_img ** 2)
        else:
            mean_img = np.zeros((H, W), dtype=np.float64)
            var_img = np.zeros((H, W), dtype=np.float64)
        
        # Applica la trasformazione alle immagini (solo per i file PNG)
        # Converti le immagini in uint8
        mean_img_uint8 = np.uint8(mean_img)
        var_img_uint8 = np.uint8(var_img)
        
        # Crea oggetti PIL per applicare flip e rotazione
        pil_mean = Image.fromarray(mean_img_uint8)
        pil_mean = pil_mean.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        pil_var = Image.fromarray(var_img_uint8)
        pil_var = pil_var.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        
        # Ottieni gli array trasformati (questi saranno salvati come PNG)
        transformed_mean = np.array(pil_mean)
        transformed_var = np.array(pil_var)
        
        # Definisci i percorsi per salvare i risultati
        mean_img_path = os.path.join(output_root, split_name, 'Mean', f'EMNIST_{split_name.lower()}_mean_class_{cls}.png')
        var_img_path = os.path.join(output_root, split_name, 'Variance', f'EMNIST_{split_name.lower()}_variance_class_{cls}.png')
        mean_csv_path = os.path.join(output_root, split_name, 'Mean', f'EMNIST_{split_name.lower()}_mean_class_{cls}.csv')
        var_csv_path = os.path.join(output_root, split_name, 'Variance', f'EMNIST_{split_name.lower()}_variance_class_{cls}.csv')
        
        # Crea le directory se non esistono
        os.makedirs(os.path.join(output_root, split_name, 'Mean'), exist_ok=True)
        os.makedirs(os.path.join(output_root, split_name, 'Variance'), exist_ok=True)
        
        # Salva l'immagine media trasformata (in scala di grigi)
        plt.figure(figsize=(4,4))
        plt.imshow(transformed_mean, cmap='gray')
        plt.axis('off')
        plt.savefig(mean_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Salva l'immagine della varianza trasformata (utilizzando il cmap 'jet')
        plt.figure(figsize=(4,4))
        plt.imshow(transformed_var, cmap='jet')
        plt.axis('off')
        plt.savefig(var_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Salva i dati in formato CSV (i dati originali, non trasformati)
        np.savetxt(mean_csv_path, mean_img, delimiter=",")
        np.savetxt(var_csv_path, var_img, delimiter=",")
        
        print(f"Processed {split_name} - Class {cls}: count = {count[cls]}")

# -------------------------
# Funzione Principale
# -------------------------
def main():
    # Imposta la cartella dei dati EMNIST (modifica in base al tuo ambiente)
    data_dir = '/home/magliolo/.cache/emnist/gzip/'
    
    # Definisci il percorso di output all'interno della cartella "results"
    output_root = os.path.join("results", "E-MNIST-PROCESSING")
    os.makedirs(output_root, exist_ok=True)
    
    # Crea le sottocartelle per il processo "Total" con "Mean" e "Variance"
    for stat in ['Mean', 'Variance']:
        os.makedirs(os.path.join(output_root, "Total", stat), exist_ok=True)
    
    # Carica i dati di EMNIST
    print("Loading training data...")
    train_images = read_idx_images(os.path.join(data_dir, 'emnist-byclass-train-images-idx3-ubyte'))
    train_labels = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-train-labels-idx1-ubyte'))
    print("Loading test data...")
    test_images = read_idx_images(os.path.join(data_dir, 'emnist-byclass-test-images-idx3-ubyte'))
    test_labels = read_idx_labels(os.path.join(data_dir, 'emnist-byclass-test-labels-idx1-ubyte'))
    
    # Combina i dati di training e test
    total_images = np.concatenate((train_images, test_images), axis=0)
    total_labels = np.concatenate((train_labels, test_labels), axis=0)
    
    # Processa e salva i risultati per il set totale
    print("Processing total dataset (Train + Test)...")
    process_dataset(total_images, total_labels, 'Total', output_root)

if __name__ == "__main__":
    main()
