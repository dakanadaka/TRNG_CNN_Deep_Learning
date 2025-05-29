# trng_cnn_checker.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import sys
import time
from datetime import datetime
import os

# Step 1: Create a custom dataset for TRNG bit sequences
class TRNGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# Step 2: Define a simple CNN for binary sequence classification
class TRNGCNN(nn.Module):
    def __init__(self):
        super(TRNGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8, 64)  # Assuming input length of 16 bits
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
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

def generate_sample_data(num_samples=1000, seq_length=16):
    """Generate random binary sequences and random labels."""
    data = np.random.randint(0, 2, (num_samples, seq_length))
    labels = np.random.randint(0, 2, num_samples)
    return data, labels

def load_real_trng_data(num_samples=1000, seq_length=16):
    """Placeholder for loading real TRNG data. Replace with actual loading logic."""
    print("[WARNING] Real TRNG data loading not implemented. Using random data as placeholder.")
    return generate_sample_data(num_samples, seq_length)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
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

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

def save_model(model, prefix):
    # Ensure the 'pth' directory exists
    os.makedirs('pth', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = f"{prefix}_{timestamp}.pth"
    filepath = os.path.join('pth', filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    return filepath

def load_model(model_path):
    model = TRNGCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="TRNG/PRNG CNN Quality Checker")
    parser.add_argument('--train', choices=['prng', 'trng'], help='Train model on PRNG or TRNG data')
    parser.add_argument('--test', choices=['prng', 'trng'], help='Test model on PRNG or TRNG data')
    parser.add_argument('--model', type=str, help='Path to model .pth file for testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seq-length', type=int, default=16, help='Sequence length')
    args = parser.parse_args()

    if not (args.train or args.test):
        print("""
Usage:
  python main.py --train=prng         # Train on simulated PRNG data
  python main.py --train=trng         # Train on (placeholder) TRNG data
  python main.py --test=prng --model=prng_YYYYMMDD_HHMMSS.pth   # Test on PRNG data
  python main.py --test=trng --model=trng_YYYYMMDD_HHMMSS.pth   # Test on TRNG data

You can adjust --epochs, --batch-size, --num-samples, --seq-length as needed.
        """)
        sys.exit(1)

    if args.train:
        if args.train == 'prng':
            data, labels = generate_sample_data(args.num_samples, args.seq_length)
            prefix = 'prng'
        else:
            data, labels = load_real_trng_data(args.num_samples, args.seq_length)
            prefix = 'trng'
        dataset = TRNGDataset(data, labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = TRNGCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, loader, criterion, optimizer, epochs=args.epochs)
        save_model(model, prefix)

    if args.test:
        if not args.model:
            print("Please provide --model path to a trained .pth file for testing.")
            sys.exit(1)
        model = load_model(args.model)
        if args.test == 'prng':
            data, labels = generate_sample_data(args.num_samples, args.seq_length)
        else:
            data, labels = load_real_trng_data(args.num_samples, args.seq_length)
        dataset = TRNGDataset(data, labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        preds = test_model(model, loader)
        print(f"Predictions on {args.test.upper()} data:", preds)

if __name__ == "__main__":
    main()
