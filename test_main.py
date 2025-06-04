import os
import numpy as np
from main import prepare_binary_data

def test_prepare_binary_data_format():
    # Setup: create small PRNG and TRNG files
    seq_length = 8
    prng_samples = ['01010101', '11110000', '00001111', '10101010']
    trng_samples = ['11001100', '00110011', '11111111', '00000000']
    os.makedirs('test_data/prng', exist_ok=True)
    os.makedirs('test_data/trng', exist_ok=True)
    prngfile = 'test_data/prng/test_prng.txt'
    trngfile = 'test_data/trng/test_trng.txt'
    with open(prngfile, 'w') as f:
        for seq in prng_samples:
            f.write(seq + '\n')
    with open(trngfile, 'w') as f:
        for seq in trng_samples:
            f.write(seq + '\n')
    # Prepare binary data
    prepare_binary_data(prngfile, trngfile, seq_length, test_size=0.5)
    # Check format of output files
    for file in ['data/binary/train.txt', 'data/binary/test.txt']:
        with open(file, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                assert len(line) == seq_length, f"Line {i} in {file} has length {len(line)}, expected {seq_length}"
                assert set(line).issubset({'0', '1'}), f"Line {i} in {file} contains invalid characters: {line}"
                assert ' ' not in line, f"Line {i} in {file} contains spaces: {line}"
    print("test_prepare_binary_data_format passed.") 