import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import os
import tempfile
import pytest
from main import (
    load_sequences_from_txt, TRNGDataset, TRNGCNN, train_model, run_inference, generate_prng_txt
)

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