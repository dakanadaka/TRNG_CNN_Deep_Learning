import torch
import torch.nn as nn
import numpy as np

# https://www.youtube.com/watch?v=HGwBXDKFk9I -> theory


# 1. Load a couple of 16-bit sequences from a txt file
# Let's assume the file 'simple_data.txt' contains:
# 1001001100110101
# 0110110011001010

def load_sequences(filepath, seq_length=16):
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == seq_length and set(line).issubset({'0', '1'}):
                sequences.append([int(bit) for bit in line])
    return np.array(sequences, dtype=np.float32)

# nn.Conv1d in PyTorch:
#   in_channels=1: Your input has 1 channel (think of this as 1 row of values — e.g., a single binary sequence like [0, 1, 1, 0, ...]).
#   out_channels=2: The layer will learn 2 different kernels, producing 2 output channels (i.e., 2 feature maps).
#   kernel_size=3: Each kernel looks at 3 consecutive values at a time — this is the receptive field of the filter.
#   padding=1: Adds 1 zero-padding on both sides, so the output length stays the same as input length.
#
# Example:
# Imagine your input is:
#   [0, 1, 1, 0, 1, 0]
# With kernel_size=3, the kernel slides like this:
#   Step 1: [0, 1, 1]
#   Step 2: [1, 1, 0]
#   Step 3: [1, 0, 1]
#   Step 4: [0, 1, 0]
# Each of these gets multiplied by the kernel weights and summed up (plus bias) → result is one output value per position, per output channel.

# 2. Define a minimal 1D CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

if __name__ == "__main__":
    # Load data
    data = load_sequences('simple_data.txt')  # shape: (num_samples, 16)
    print("Loaded data:")
    print(data)

    # Convert to tensor and add channel dimension
    tensor = torch.tensor(data).unsqueeze(1)  # shape: (num_samples, 1, 16)
    print("\nTensor shape:", tensor.shape)
    print(tensor)

    # Create and run the simple model
    model = SimpleCNN()
    output = model(tensor)
    print("\nOutput shape:", output.shape)
    print(output) 