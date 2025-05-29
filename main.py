"""
TRNG/PRNG CNN Quality Checker
-----------------------------
This script provides a command-line tool for training and evaluating a 1D Convolutional Neural Network (CNN)
on binary sequences, such as those produced by True Random Number Generators (TRNGs) or Pseudo-Random Number Generators (PRNGs).

Why a CNN?
-----------
A 1D CNN is well-suited for detecting local patterns and dependencies in sequential data, such as bitstreams.
By applying convolutional filters, the model can learn to recognize features that distinguish random from non-random sequences.

Features:
- Generate or load binary sequence data (from .txt files)
- Train a CNN to classify sequences
- Save/load models
- Run inference on new data
- Fully configurable via CLI arguments

Usage examples:
  python main.py --generate-prng data/prng/generated.txt --num-samples=1000 --seq-length=16
  python main.py --train=prng --datafile=data/prng/generated.txt
  python main.py --test=prng --model=pth/prng_YYYYMMDD_HHMMSS.pth --datafile=data/prng/generated.txt

See README.md for more details.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import sys
import time
import os

# -----------------------------
# Dataset for binary sequences
# -----------------------------
class TRNGDataset(Dataset):
    """
    PyTorch Dataset for binary sequences and labels.
    Args:
        data (np.ndarray): Array of shape (N, seq_length) with 0/1 values.
        labels (np.ndarray): Array of shape (N,) with integer labels (0 or 1).
    Returns:
        (torch.Tensor, torch.Tensor): Sequence and label for each sample.
    """
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# -----------------------------
# 1D CNN Model for Sequence Classification
# -----------------------------
class TRNGCNN(nn.Module):
    """
    1D Convolutional Neural Network for binary sequence classification.
    The architecture consists of two Conv1d layers, ReLU activations, MaxPool, and two fully connected layers.
    The input size of the first FC layer is computed dynamically based on the input sequence length.
    Args:
        seq_length (int): Length of the input sequences.
    """
    def __init__(self, seq_length=16):
        super(TRNGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        # Dynamically compute the flattened size after conv/pool layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_length)
            x = self.conv1(dummy)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            flat_size = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        """
        Forward pass for the CNN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, seq_length)
        Returns:
            torch.Tensor: Output logits of shape (batch, 2)
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# -----------------------------
# Data Loading Utilities
# -----------------------------
def load_sequences_from_txt(filepath, seq_length):
    """
    Load binary sequences from a .txt file. Each line must be a string of 0s and 1s of length seq_length.
    Args:
        filepath (str): Path to the .txt file.
        seq_length (int): Expected length of each sequence.
    Returns:
        np.ndarray: Array of shape (N, seq_length) with 0/1 values.
    Raises:
        ValueError: If any line is invalid.
    """
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) != seq_length or not set(line).issubset({'0', '1'}):
                raise ValueError(f"Invalid line in {filepath}: '{line}' (must be {seq_length} bits of 0/1)")
            sequences.append([int(bit) for bit in line])
    return np.array(sequences, dtype=np.float32)

def generate_prng_txt(filename, num_samples, seq_length):
    """
    Generate a .txt file with random PRNG sequences (0s and 1s).
    Args:
        filename (str): Output file path.
        num_samples (int): Number of sequences to generate.
        seq_length (int): Length of each sequence.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for _ in range(num_samples):
            seq = ''.join(str(np.random.randint(0, 2)) for _ in range(seq_length))
            f.write(seq + '\n')
    print(f"Generated {num_samples} PRNG sequences in {filename}")

# -----------------------------
# Training and Inference
# -----------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """
    Train the CNN model on the provided data.
    Args:
        model (nn.Module): The CNN model.
        train_loader (DataLoader): DataLoader for training data.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        optimizer: Optimizer (e.g., torch.optim.Adam).
        epochs (int): Number of training epochs.
    Prints:
        Training loss per epoch.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def run_inference(model, test_loader):
    """
    Run inference on a dataset using the trained model.
    Args:
        model (nn.Module): Trained CNN model.
        test_loader (DataLoader): DataLoader for test data.
    Returns:
        np.ndarray: Predicted class labels for each input sequence.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

# -----------------------------
# Model Saving/Loading
# -----------------------------
def save_model(model, prefix):
    """
    Save the model's state_dict to a timestamped .pth file in the 'pth/' directory.
    Args:
        model (nn.Module): The trained model.
        prefix (str): Prefix for the filename (e.g., 'prng' or 'trng').
    Returns:
        str: Path to the saved file.
    """
    # Ensure the 'pth' directory exists
    os.makedirs('pth', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = f"{prefix}_{timestamp}.pth"
    filepath = os.path.join('pth', filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    return filepath

def load_model(model_path, seq_length=16):
    """
    Load a model from a .pth file.
    Args:
        model_path (str): Path to the .pth file.
        seq_length (int): Sequence length for model architecture.
    Returns:
        nn.Module: Loaded model in eval mode.
    """
    model = TRNGCNN(seq_length=seq_length)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# -----------------------------
# Command-Line Interface (CLI)
# -----------------------------
def main():
    """
    Main CLI entry point. Parses arguments and runs the requested action:
    - Generate PRNG data
    - Train a model
    - Test a model
    """
    parser = argparse.ArgumentParser(description="TRNG/PRNG CNN Quality Checker")
    parser.add_argument('--train', choices=['prng', 'trng'], help='Train model on PRNG or TRNG data')
    parser.add_argument('--test', choices=['prng', 'trng'], help='Test model on PRNG or TRNG data')
    parser.add_argument('--model', type=str, help='Path to model .pth file for testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples (ignored if datafile is used)')
    parser.add_argument('--seq-length', type=int, default=16, help='Sequence length')
    parser.add_argument('--datafile', type=str, help='Path to .txt file with binary sequences (required for trng/prng data)')
    parser.add_argument('--generate-prng', type=str, help='Generate PRNG .txt file at given path (e.g., data/prng/generated.txt)')
    args = parser.parse_args()

    # Generate PRNG data and exit
    if args.generate_prng:
        generate_prng_txt(args.generate_prng, args.num_samples, args.seq_length)
        sys.exit(0)

    # Print usage if no action specified
    if not (args.train or args.test):
        print("""
Usage:
  python main.py --generate-prng data/prng/generated.txt --num-samples=1000 --seq-length=16
  python main.py --train=prng --datafile=data/prng/yourfile.txt
  python main.py --test=prng --model=pth/yourmodel.pth --datafile=data/prng/yourfile.txt
  python main.py --train=trng --datafile=data/trng/yourfile.txt
  python main.py --test=trng --model=pth/yourmodel.pth --datafile=data/trng/yourfile.txt

You can adjust --epochs, --batch-size, --seq-length as needed.
        """)
        sys.exit(1)

    # Datafile checks for PRNG/TRNG
    if (args.train == 'trng' or args.test == 'trng'):
        if not args.datafile or not args.datafile.startswith('data/trng/'):
            print('For trng, you must provide --datafile=data/trng/yourfile.txt')
            sys.exit(1)
    if (args.train == 'prng' or args.test == 'prng'):
        if not args.datafile or not args.datafile.startswith('data/prng/'):
            print('For prng, you must provide --datafile=data/prng/yourfile.txt')
            sys.exit(1)

    # Training
    if args.train:
        if args.datafile:
            try:
                data = load_sequences_from_txt(args.datafile, args.seq_length)
                labels = np.zeros(len(data), dtype=np.int64) if args.train == 'prng' else np.ones(len(data), dtype=np.int64)
            except Exception as e:
                print(f"Error loading datafile: {e}")
                sys.exit(1)
            prefix = args.train
        else:
            data, labels = generate_sample_data(args.num_samples, args.seq_length)
            prefix = 'prng'
        dataset = TRNGDataset(data, labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = TRNGCNN(seq_length=args.seq_length)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, loader, criterion, optimizer, epochs=args.epochs)
        save_model(model, prefix)

    # Testing
    if args.test:
        if not args.model:
            print("Please provide --model path to a trained .pth file for testing.")
            sys.exit(1)
        model = load_model(args.model, seq_length=args.seq_length)
        if args.datafile:
            try:
                data = load_sequences_from_txt(args.datafile, args.seq_length)
                labels = np.zeros(len(data), dtype=np.int64) if args.test == 'prng' else np.ones(len(data), dtype=np.int64)
            except Exception as e:
                print(f"Error loading datafile: {e}")
                sys.exit(1)
        else:
            data, labels = generate_sample_data(args.num_samples, args.seq_length)
        dataset = TRNGDataset(data, labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        preds = run_inference(model, loader)
        print(f"Predictions on {args.test.upper()} data:", preds)

# -----------------------------
# Data Generation for Simulated Data (if not using files)
# -----------------------------
def generate_sample_data(num_samples=1000, seq_length=16):
    """
    Generate random binary sequences and random labels for prototyping.
    Args:
        num_samples (int): Number of samples to generate.
        seq_length (int): Length of each sequence.
    Returns:
        (np.ndarray, np.ndarray): Data and labels arrays.
    """
    data = np.random.randint(0, 2, (num_samples, seq_length))
    labels = np.random.randint(0, 2, num_samples)
    return data, labels

if __name__ == "__main__":
    main()
