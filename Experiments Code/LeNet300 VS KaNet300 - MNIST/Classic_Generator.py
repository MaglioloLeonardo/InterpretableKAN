import os
import inspect
import argparse
import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
import pandas as pd
import matplotlib.pyplot as plt
from architectures_28x28.KANLinearFullyConnected import KANLinearFullyConnected 

# Parsing degli argomenti da linea di comando
parser = argparse.ArgumentParser(description="Parametri per l'addestramento del modello")
parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate per l'ottimizzatore")
parser.add_argument("--norm_type", choices=["L1", "L2", "None"], required=True, help="Tipologia di norma (L1, L2 o None)")
parser.add_argument("--optimizer_type", choices=["SGD", "Adam"], required=True, help="Tipo di ottimizzatore (SGD o Adam)")
parser.add_argument("--batch_size", type=int, required=True, help="Dimensione del batch")
parser.add_argument("--epochs", type=int, required=True, help="Numero di epoche di addestramento")
parser.add_argument("--grid_size", type=int, required=True, help="Dimensione della griglia per il modello KANLinearFullyConnected")
parser.add_argument("--spline_order", type=int, required=True, help="Ordine dello spline per il modello KANLinearFullyConnected")

args = parser.parse_args()

# Impostazione dei parametri dal parser
learning_rate = args.learning_rate
norm_type = args.norm_type
optimizer_type = args.optimizer_type
batch_size = args.batch_size
epochs = args.epochs
grid_size = args.grid_size
spline_order = args.spline_order

# Set the seed for reproducibility
seed = 12
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Define the transformations for MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Creazione del train e validation set (45k per training, 5k per validation) in modo deterministico
'''train_indices = list(range(0, 45000))
val_indices = list(range(45000, 50000))
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)'''

# Definizione delle dimensioni per training e validation set
val_size = 5000
train_size = len(full_train_dataset) - val_size  # 60.000 - 5.000 = 55.000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print(f"Dimensione del training set: {len(train_dataset)}")
print(f"Dimensione del validation set: {len(val_dataset)}")
print(f"Dimensione del test set: {len(test_dataset)}")


class LeNet300_OutputOnly(nn.Module):
    """LeNet-300 model with only output layer."""
    def __init__(self):
        super(LeNet300_OutputOnly, self).__init__()
        self.fc = nn.Linear(784, 10)  # Only input layer and output layer
        self.relu = nn.ReLU()         # Define ReLU explicitly

    def forward(self, x):
        x = x.view(x.size(0), -1)    # Flatten the MNIST image (28x28 -> 784)
        x = self.fc(x)               # Linear transformation
        return x  

class LeNet300_HiddenLayerOnly(nn.Module):
    """LeNet-300 model with only the 300 hidden layer and output layer."""
    def __init__(self):
        super(LeNet300_HiddenLayerOnly, self).__init__()
        self.fc1 = nn.Linear(784, 300)  # Input layer -> 300 hidden layer
        self.fc2 = nn.Linear(300, 10)   # 300 hidden layer -> Output layer
        self.relu = nn.ReLU()           # Define ReLU explicitly

    def forward(self, x):
        x = x.view(x.size(0), -1)     # Flatten the MNIST image (28x28 -> 784)
        x = self.fc1(x)               # Linear transformation for hidden layer
        x = self.relu(x)              # ReLU activation for hidden layer
        x = self.fc2(x)               # Linear transformation for output layer
        return x  


class LeNet300_Full(nn.Module):
    """Full LeNet-300-100 model."""
    def __init__(self):
        super(LeNet300_Full, self).__init__()
        self.fc1 = nn.Linear(784, 300)  # Input layer -> 300 hidden layer
        self.fc2 = nn.Linear(300, 100)  # 300 hidden layer -> 100 hidden layer
        self.fc3 = nn.Linear(100, 10)   # 100 hidden layer -> Output layer
        self.relu = nn.ReLU()           # Define ReLU explicitly

    def forward(self, x):
        x = x.view(x.size(0), -1)     # Flatten the MNIST image (28x28 -> 784)
        x = self.fc1(x)               # Linear transformation for first hidden layer
        x = self.relu(x)              # ReLU activation for first hidden layer
        x = self.fc2(x)               # Linear transformation for second hidden layer
        x = self.relu(x)              # ReLU activation for second hidden layer
        x = self.fc3(x)               # Linear transformation for output layer
        return x 

def save_script_to_directory(notebook_name, directory=None, filename="saved_script.py"):
        # Determine the current folder or use the provided directory
    if directory is None:
        current_folder = globals()['_dh'][0]  # Get the current notebook directory
    else:
        current_folder = directory

    notebook_path = os.path.join(current_folder, notebook_name)

    try:
        # Read the notebook content
        with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
            notebook_content = notebook_file.read()

        # Save the notebook content as a .py file in the specified directory
        script_path = os.path.join(current_folder, filename)
        with open(script_path, 'w', encoding='utf-8') as script_file:
            script_file.write(notebook_content)
        
        print(f"Notebook saved as script to {script_path}")

    except FileNotFoundError:
        print(f"Notebook not found at {notebook_path}")


def generate_basedir_name(optimizer, grid_size, spline_order, lr, norm_type):
    # Extract the optimizer name
    optimizer_name = optimizer.__class__.__name__

    # Use 'none' if no norm type is provided
    norm_suffix = norm_type #if norm_type else "none"

    # Construct the base directory name
    base_dir = f"results_standard_{norm_suffix}_{optimizer_name}_lr{lr}_0_0"
    return base_dir

def save_params_to_file(base_dir, args):
    params_file = os.path.join(base_dir, "params.txt")
    with open(params_file, "w") as f:
        f.write("Parametri di esecuzione:\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Norm Type: {args.norm_type}\n")
        f.write(f"Optimizer Type: {args.optimizer_type}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Grid Size: {args.grid_size}\n")
        f.write(f"Spline Order: {args.spline_order}\n")
    print(f"Parametri salvati in {params_file}")
    
# Function to save the model parameters and description to a text file
def save_model_description(base_dir, model_name, layers_hidden, optimizer_params, description, scheduler_params, batch_size):
    file_path = os.path.join(base_dir, 'model_description.txt')
    with open(file_path, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write("Model Architecture:\n")
        f.write(f"  - Layers: {layers_hidden}\n")
        f.write(f"Optimizer Parameters:\n")
        f.write(f"  - Optimizer: SGD (no momentum)\n")
        f.write(f"  - Learning Rate: {optimizer_params['lr']}\n")
        f.write(f"  - Weight Decay: {optimizer_params['weight_decay']}\n")
        f.write(f"Scheduler Parameters:\n")
        f.write(f"  - Scheduler: ExponentialLR\n")
        f.write(f"  - Gamma: {scheduler_params['gamma']}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Grid Size: {grid_size}\n")
        f.write(f"Spline order: {spline_order}\n")
        
        f.write("\nModel Description:\n")
        f.write(f"{description}\n")

# Function to save confusion matrices
def save_confusion_matrix(cm, file_path_image, file_path_csv, title='Confusion Matrix'):
    # Save as image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(file_path_image)
    plt.close()

    # Save as CSV
    pd.DataFrame(cm).to_csv(file_path_csv, index=False)

# Function to normalize confusion matrix
def normalize_confusion_matrix(cm):
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

def save_losses_to_csv(base_dir, train_losses, val_losses, test_losses, epochs):
    file_path = os.path.join(base_dir, 'losses.csv')
    df = pd.DataFrame({
        'Epoch': list(range(1, epochs + 1)),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Test Loss': test_losses
    })
    df.to_csv(file_path, index=False)


# Function to train and evaluate each model
def train_and_evaluate(model, model_name, layers_hidden, description, lr, epochs, norm_type, grid_size, spline_order):
    # Define the loss function and optimizer (SGD without momentum)
    criterion = nn.CrossEntropyLoss()
    gamma = 0.96
    momentum = 0
    weight_decay = 0
    if norm_type == "L1":
        weight_decay = 0
        l1_lambda = 0.0001  # This is the L1 regularization term
    elif norm_type == "L2":
        weight_decay = 0.001

    print(weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)  # Using SGD without momentum
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Save model description and parameters
    optimizer_params = {'lr': lr, 'weight_decay': optimizer.defaults['weight_decay']}
    scheduler_params = {'gamma': gamma}
    
    # Create directories for saving model checkpoints and results
    base_dir = os.path.join('results', os.path.join(generate_basedir_name(optimizer, grid_size, spline_order, lr, norm_type), model_name))
    os.makedirs(base_dir, exist_ok=True)
    folders = {
        'normalized_img': os.path.join(base_dir, 'confusion_matrices_normalized_img'),
        'normalized_csv': os.path.join(base_dir, 'confusion_matrices_normalized_csv'),
        'not_normalized_img': os.path.join(base_dir, 'confusion_matrices_not_normalized_img'),
        'not_normalized_csv': os.path.join(base_dir, 'confusion_matrices_not_normalized_csv'),
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    save_model_description(base_dir, model_name, layers_hidden, optimizer_params, description, scheduler_params, batch_size)
   # save_script_to_directory(notebook_name = "EsperimentiKanCorretti.ipynb", directory = base_dir)
    
    # Training loop
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []


    for epoch in range(epochs):
        # Reset running loss for training
        running_loss_train = 0.0

        # Train over entire training set (batch by batch) - ONLY OPTIMIZATION
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if norm_type == "L1":
                loss += l1_lambda * sum(torch.sum(torch.abs(param)) for param in model.parameters())  # Aggiungi la penalità L1
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()

        # Calculate confusion matrix and metrics for training set at the end of the epoch
        model.eval()
        with torch.no_grad():
            all_train_preds = []
            all_train_targets = []
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred_train = output.argmax(dim=1, keepdim=False)
                all_train_preds.append(pred_train.cpu().numpy())
                all_train_targets.append(target.cpu().numpy())

        # Concatenate all predictions and targets
        all_train_preds = np.concatenate(all_train_preds)
        all_train_targets = np.concatenate(all_train_targets)
        full_train_conf_matrix = confusion_matrix(all_train_targets, all_train_preds, labels=range(10))

        # Calculate accuracy and loss for training
        train_accuracy = np.trace(full_train_conf_matrix) / len(train_loader.dataset)
        train_loss = running_loss_train / len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Reset running loss for test
        running_loss_test = 0.0

        # Test phase over entire test set (batch by batch)
        all_test_preds = []
        all_test_targets = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                if norm_type == "L1":
                    loss += l1_lambda * sum(torch.sum(torch.abs(param)) for param in model.parameters())  # Aggiungi la penalità L1
                running_loss_test += loss.item()

                # Collect predictions and targets for later metric computation
                pred_test = output.argmax(dim=1, keepdim=False)
                all_test_preds.append(pred_test.cpu().numpy())
                all_test_targets.append(target.cpu().numpy())

        # Concatenate all predictions and targets
        all_test_preds = np.concatenate(all_test_preds)
        all_test_targets = np.concatenate(all_test_targets)
        full_test_conf_matrix = confusion_matrix(all_test_targets, all_test_preds, labels=range(10))

        # Calculate accuracy and loss for test set
        test_accuracy = np.trace(full_test_conf_matrix) / len(test_loader.dataset)
        test_loss = running_loss_test / len(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Calcolo della loss e accuracy per il validation set
        running_loss_val = 0.0
        all_val_preds = []
        all_val_targets = []
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                if norm_type == "L1":
                    loss += l1_lambda * sum(torch.sum(torch.abs(param)) for param in model.parameters())  # Aggiungi la penalità L1
                running_loss_val += loss.item()
        
                # Raccolta delle predizioni e dei target per calcolare la confusion matrix
                pred_val = output.argmax(dim=1, keepdim=False)
                all_val_preds.append(pred_val.cpu().numpy())
                all_val_targets.append(target.cpu().numpy())
        
        # Concatenate le predizioni e i target
        all_val_preds = np.concatenate(all_val_preds)
        all_val_targets = np.concatenate(all_val_targets)
        full_val_conf_matrix = confusion_matrix(all_val_targets, all_val_preds, labels=range(10))
        
        # Calcolo dell'accuracy e della loss per il validation set
        val_accuracy = np.trace(full_val_conf_matrix) / len(val_loader.dataset)
        val_loss = running_loss_val / len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


        
        # Print the results for the epoch
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Step the learning rate scheduler
        scheduler.step()

        # Save confusion matrices for both train and test (non-normalized and normalized)
        save_confusion_matrix(
            full_train_conf_matrix,
            os.path.join(folders['not_normalized_img'], f'train_conf_matrix_epoch_{epoch + 1}.png'),
            os.path.join(folders['not_normalized_csv'], f'train_conf_matrix_epoch_{epoch + 1}.csv'),
            title=f'Train Confusion Matrix - Epoch {epoch + 1}'
        )
        save_confusion_matrix(
            full_test_conf_matrix,
            os.path.join(folders['not_normalized_img'], f'test_conf_matrix_epoch_{epoch + 1}.png'),
            os.path.join(folders['not_normalized_csv'], f'test_conf_matrix_epoch_{epoch + 1}.csv'),
            title=f'Test Confusion Matrix - Epoch {epoch + 1}'
        )

        # Save normalized confusion matrices
        norm_train_conf_matrix = normalize_confusion_matrix(full_train_conf_matrix)
        save_confusion_matrix(
            norm_train_conf_matrix,
            os.path.join(folders['normalized_img'], f'norm_train_conf_matrix_epoch_{epoch + 1}.png'),
            os.path.join(folders['normalized_csv'], f'norm_train_conf_matrix_epoch_{epoch + 1}.csv'),
            title=f'Normalized Train Confusion Matrix - Epoch {epoch + 1}'
        )
        norm_test_conf_matrix = normalize_confusion_matrix(full_test_conf_matrix)
        save_confusion_matrix(
            norm_test_conf_matrix,
            os.path.join(folders['normalized_img'], f'norm_test_conf_matrix_epoch_{epoch + 1}.png'),
            os.path.join(folders['normalized_csv'], f'norm_test_conf_matrix_epoch_{epoch + 1}.csv'),
            title=f'Normalized Test Confusion Matrix - Epoch {epoch + 1}'
        )


        # Salvataggio delle confusion matrix per il validation set (non normalizzata e normalizzata)
        save_confusion_matrix(
            full_val_conf_matrix,
            os.path.join(folders['not_normalized_img'], f'val_conf_matrix_epoch_{epoch + 1}.png'),
            os.path.join(folders['not_normalized_csv'], f'val_conf_matrix_epoch_{epoch + 1}.csv'),
            title=f'Validation Confusion Matrix - Epoch {epoch + 1}'
        )
        
        # Normalizzazione della confusion matrix del validation set
        norm_val_conf_matrix = normalize_confusion_matrix(full_val_conf_matrix)
        save_confusion_matrix(
            norm_val_conf_matrix,
            os.path.join(folders['normalized_img'], f'norm_val_conf_matrix_epoch_{epoch + 1}.png'),
            os.path.join(folders['normalized_csv'], f'norm_val_conf_matrix_epoch_{epoch + 1}.csv'),
            title=f'Normalized Validation Confusion Matrix - Epoch {epoch + 1}'
        )


        # Save checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            torch.save(checkpoint, os.path.join(model_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    # Export the loss values to CSV
    save_losses_to_csv(base_dir, train_losses, val_losses, test_losses, epochs)

    # Grafico dell'accuracy per training, validation e test set
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Training, Validation, and Test Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, 'accuracy_per_epoch.png'))
    plt.close()
    
    # Grafico della loss per training, validation e test set
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training, Validation, and Test Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, 'loss_per_epoch.png'))
    plt.close()

    
    #save_params_to_file("results/" +generate_basedir_name(optimizer_type, grid_size, spline_order, learning_rate, norm_type), args)


# Train and evaluate the three configurations
model_output_only = LeNet300_OutputOnly().to(device)
train_and_evaluate(model_output_only, 'LeNet300_OutputOnly', [784, 10], "This model uses only the output layer.", learning_rate, epochs, norm_type,  grid_size, spline_order)

model_hidden_only = LeNet300_HiddenLayerOnly().to(device)
train_and_evaluate(model_hidden_only, 'LeNet300_HiddenLayerOnly', [784, 300, 10], "This model uses a single hidden layer with 300 neurons.", learning_rate, epochs, norm_type,  grid_size, spline_order)

model_full = LeNet300_Full().to(device)
train_and_evaluate(model_full, 'LeNet300_Full', [784, 300, 100, 10], "This is the full LeNet-300-100 architecture.", learning_rate, epochs, norm_type,  grid_size, spline_order)

print("Training complete.")