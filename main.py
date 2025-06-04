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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class DeviceManager:
    """
    Utility class to manage device selection (CUDA or CPU) for PyTorch.
    Usage:
        device_manager = DeviceManager()
        device = device_manager.device
    """
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device('cpu')
            print("Using device: CPU")

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
        for i in range(num_samples):
            seq = ''.join(str(np.random.randint(0, 2)) for _ in range(seq_length))
            f.write(seq + '\n')
            if (i + 1) % 1_000_000 == 0:
                percent = 100 * (i + 1) / num_samples
                print(f"Generated {i + 1:,} / {num_samples:,} sequences... ({percent:.1f}%)", flush=True)
    print(f"Generated {num_samples} PRNG sequences in {filename}")

def generate_trng_txt(filename, num_samples, seq_length):
    """
    Generate a .txt file with fake TRNG sequences (0s and 1s).
    Args:
        filename (str): Output file path.
        num_samples (int): Number of sequences to generate.
        seq_length (int): Length of each sequence.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for i in range(num_samples):
            seq = ''.join(str(np.random.randint(0, 2)) for _ in range(seq_length))
            f.write(seq + '\n')
            if (i + 1) % 1_000_000 == 0:
                percent = 100 * (i + 1) / num_samples
                print(f"Generated {i + 1:,} / {num_samples:,} sequences... ({percent:.1f}%)", flush=True)
    print(f"Generated {num_samples} TRNG sequences in {filename}", flush=True)

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
    parser.add_argument('--generate-trng', type=str, help='Generate TRNG .txt file at given path (e.g., data/trng/generated.txt)')
    parser.add_argument('--split-data', nargs=3, metavar=('INPUT', 'TRAIN_OUT', 'TEST_OUT'), help='Split a .txt file into train/test sets: --split-data input.txt train.txt test.txt')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test set fraction for --split-data (default 0.3)')
    parser.add_argument('--metrics', action='store_true', help='Print accuracy and confusion matrix after testing (requires true labels in datafile)')
    parser.add_argument('--labelsfile', type=str, help='Path to .txt file with labels (one per line, 0 or 1)')
    parser.add_argument('--prepare-binary-data', action='store_true', help='Prepare binary classification data from PRNG and TRNG files')
    parser.add_argument('--prngfile', type=str, help='Path to PRNG .txt file for binary data preparation')
    parser.add_argument('--trngfile', type=str, help='Path to TRNG .txt file for binary data preparation')
    args = parser.parse_args()

    # Generate PRNG data and exit
    if args.generate_prng:
        generate_prng_txt(args.generate_prng, args.num_samples, args.seq_length)
        sys.exit(0)

    # Generate TRNG data and exit
    if args.generate_trng:
        generate_trng_txt(args.generate_trng, args.num_samples, args.seq_length)
        sys.exit(0)

    # Data splitting utility
    if args.split_data:
        input_file, train_file, test_file = args.split_data
        split_txt_file(input_file, train_file, test_file, test_size=args.test_size, seq_length=args.seq_length)
        sys.exit(0)

    # Prepare binary classification data and exit
    if args.prepare_binary_data:
        if not args.prngfile or not args.trngfile:
            print('Please provide --prngfile and --trngfile for binary data preparation.')
            sys.exit(1)
        prepare_binary_data(args.prngfile, args.trngfile, args.seq_length, test_size=args.test_size)
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

    # Training
    if args.train:
        if args.datafile:
            try:
                data = load_sequences_from_txt(args.datafile, args.seq_length)
                if args.labelsfile:
                    labels = np.loadtxt(args.labelsfile, dtype=np.int64)
                    if len(labels) != len(data):
                        print("Error: Number of labels does not match number of data samples.")
                        sys.exit(1)
                else:
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
        device_manager = DeviceManager()
        device = device_manager.device
        print(f"Training on {device}")
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
                if args.labelsfile:
                    labels = np.loadtxt(args.labelsfile, dtype=np.int64)
                    if len(labels) != len(data):
                        print("Error: Number of labels does not match number of data samples.")
                        sys.exit(1)
                else:
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
        # If --metrics is set, try to load true labels and print accuracy/confusion matrix
        if args.metrics:
            try:
                if args.labelsfile:
                    true_labels = labels
                else:
                    true_label = 0 if args.test == 'prng' else 1
                    true_labels = np.full(len(preds), true_label, dtype=np.int64)
                acc = accuracy_score(true_labels, preds)
                cm = confusion_matrix(true_labels, preds)
                print(f"Accuracy: {acc:.4f}")
                print("Confusion Matrix:\n", cm)
            except Exception as e:
                print(f"Could not compute metrics: {e}")

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

def split_txt_file(input_file, train_file, test_file, test_size=0.3, seq_length=16, random_state=42):
    """
    Split a .txt file of binary sequences into train and test files.
    Args:
        input_file (str): Path to input .txt file.
        train_file (str): Output path for train set.
        test_file (str): Output path for test set.
        test_size (float): Fraction of data to use for test set.
        seq_length (int): Sequence length for validation.
        random_state (int): Random seed for reproducibility.
    """
    # Load all sequences
    data = load_sequences_from_txt(input_file, seq_length)
    # For splitting, generate dummy labels (not used for actual training)
    labels = np.zeros(len(data), dtype=np.int64)
    X_train, X_test, _, _ = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    # Save train and test sets
    with open(train_file, 'w') as f:
        for seq in X_train:
            f.write(''.join(str(int(bit)) for bit in seq) + '\n')
    with open(test_file, 'w') as f:
        for seq in X_test:
            f.write(''.join(str(int(bit)) for bit in seq) + '\n')
    print(f"Split {input_file} into {len(X_train)} train and {len(X_test)} test sequences.")

def prepare_binary_data(prngfile, trngfile, seq_length, test_size=0.3):
    """
    Combine PRNG and TRNG .txt files, label, shuffle, and split into train/test sets for binary classification.
    Args:
        prngfile (str): Path to PRNG .txt file.
        trngfile (str): Path to TRNG .txt file.
        seq_length (int): Sequence length.
        test_size (float): Fraction for test set.
    Outputs:
        data/binary/train.txt, data/binary/train_labels.txt, data/binary/test.txt, data/binary/test_labels.txt
    """
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    # Load PRNG
    with open(prngfile, 'r') as f:
        prng = [line.strip() for line in f if line.strip()]
    prng_data = np.array([list(map(int, seq)) for seq in prng])
    prng_labels = np.zeros(len(prng_data), dtype=np.int64)
    # Load TRNG
    with open(trngfile, 'r') as f:
        trng = [line.strip() for line in f if line.strip()]
    trng_data = np.array([list(map(int, seq)) for seq in trng])
    trng_labels = np.ones(len(trng_data), dtype=np.int64)
    # Combine and shuffle
    data = np.vstack([prng_data, trng_data])
    labels = np.concatenate([prng_labels, trng_labels])
    indices = np.random.permutation(len(data))
    data, labels = data[indices], labels[indices]
    # Split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    # Save to files
    os.makedirs('data/binary', exist_ok=True)
    np.savetxt('data/binary/train.txt', [''.join(map(str, row)) for row in X_train], fmt='%s')
    np.savetxt('data/binary/test.txt', [''.join(map(str, row)) for row in X_test], fmt='%s')
    np.savetxt('data/binary/train_labels.txt', y_train, fmt='%d')
    np.savetxt('data/binary/test_labels.txt', y_test, fmt='%d')
    print('Prepared binary classification data:')
    print(f'  Train: {X_train.shape[0]} samples')
    print(f'  Test:  {X_test.shape[0]} samples')

# -----------------------------
# Test for data format
# -----------------------------
def test_prepare_binary_data_format(train_file, test_file, seq_length):
    """
    Test that each line in the train and test files is a string of 0s and 1s of the correct length, with no spaces.
    """
    for file in [train_file, test_file]:
        with open(file, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                assert len(line) == seq_length, f"Line {i} in {file} has length {len(line)}, expected {seq_length}"
                assert set(line).issubset({'0', '1'}), f"Line {i} in {file} contains invalid characters: {line}"
                assert ' ' not in line, f"Line {i} in {file} contains spaces: {line}"
    print(f"Format test passed for {train_file} and {test_file}")

if __name__ == "__main__":
    main()
