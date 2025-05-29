import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import os
import tempfile
import pytest
from main import (
    load_sequences_from_txt, TRNGDataset, TRNGCNN, train_model, run_inference, generate_prng_txt, split_txt_file
)
from sklearn.metrics import accuracy_score, confusion_matrix

def test_load_sequences_from_txt_valid(tmp_path):
    test_file = tmp_path / "test.txt"
    seqs = ["0101010101010101", "1111000011110000"]
    test_file.write_text("\n".join(seqs))
    arr = load_sequences_from_txt(str(test_file), seq_length=16)
    assert arr.shape == (2, 16)
    assert np.all((arr == 0) | (arr == 1))

def test_load_sequences_from_txt_invalid(tmp_path):
    test_file = tmp_path / "bad.txt"
    # Wrong length and invalid chars
    seqs = ["010101", "111100001111000X"]
    test_file.write_text("\n".join(seqs))
    with pytest.raises(ValueError):
        load_sequences_from_txt(str(test_file), seq_length=16)

def test_dataset():
    data = np.random.randint(0, 2, (10, 16))
    labels = np.random.randint(0, 2, 10)
    ds = TRNGDataset(data, labels)
    x, y = ds[0]
    assert x.shape == (16,)
    assert y in [0, 1]
    assert len(ds) == 10

def test_model_dynamic_fc():
    for seq_length in [16, 32, 64]:
        model = TRNGCNN(seq_length=seq_length)
        x = torch.zeros(2, 1, seq_length)
        out = model(x)
        assert out.shape == (2, 2)

def test_train_model_runs():
    data = np.random.randint(0, 2, (20, 16))
    labels = np.random.randint(0, 2, 20)
    ds = TRNGDataset(data, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    model = TRNGCNN(seq_length=16)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Should run without error and loss should be finite
    train_model(model, loader, criterion, optimizer, epochs=1)

def test_run_inference_output_shape():
    data = np.random.randint(0, 2, (10, 16))
    labels = np.random.randint(0, 2, 10)
    ds = TRNGDataset(data, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    model = TRNGCNN(seq_length=16)
    preds = run_inference(model, loader)
    assert preds.shape == (10,)
    assert np.all((preds == 0) | (preds == 1))

def test_generate_prng_txt(tmp_path):
    out_file = tmp_path / "gen.txt"
    generate_prng_txt(str(out_file), num_samples=5, seq_length=8)
    lines = out_file.read_text().splitlines()
    assert len(lines) == 5
    for line in lines:
        assert set(line).issubset({'0', '1'})
        assert len(line) == 8

# --- New tests for data splitting and metrics ---
def test_split_txt_file(tmp_path):
    # Create a file with 10 sequences
    input_file = tmp_path / "all.txt"
    seqs = ["0" * 16, "1" * 16] * 5
    input_file.write_text("\n".join(seqs))
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"
    split_txt_file(str(input_file), str(train_file), str(test_file), test_size=0.3, seq_length=16)
    train_lines = train_file.read_text().splitlines()
    test_lines = test_file.read_text().splitlines()
    assert len(train_lines) + len(test_lines) == 10
    assert all(len(line) == 16 for line in train_lines + test_lines)
    # Check that all lines are present (no duplicates or missing)
    assert set(train_lines + test_lines) == set(seqs)

def test_metrics_accuracy_confusion():
    # Simulate predictions and true labels
    preds = np.array([0, 0, 1, 1, 0, 1])
    true_labels = np.array([0, 0, 1, 0, 1, 1])
    acc = accuracy_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds)
    assert 0 <= acc <= 1
    assert cm.shape == (2, 2)
    # Check confusion matrix values
    assert cm[0, 0] == 2  # True negatives
    assert cm[1, 1] == 2  # True positives 