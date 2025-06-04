# TRNG/PRNG CNN Quality Checker

> **Based on:** [Assessing the quality of random number generators through neural networks](https://www.researchgate.net/publication/381370870_Assessing_the_quality_of_random_number_generators_through_neural_networks)
>
> This project is inspired by and implements the approach described in the above research paper, which demonstrates how neural networks (specifically 1D CNNs) can be used to assess the quality of random number generators by detecting patterns in bitstreams.

This tool provides a command-line interface for training and evaluating a 1D Convolutional Neural Network (CNN) on binary sequences, such as those produced by True Random Number Generators (TRNGs) or Pseudo-Random Number Generators (PRNGs).

## Features
- Generate or load binary sequence data (from .txt files)
- Train a CNN to classify sequences
- Save/load models
- Run inference on new data
- Fully configurable via CLI arguments
- Supports binary classification (PRNG vs TRNG) with custom label files

## Usage Examples

### 1. Generate PRNG and TRNG Data
Generate PRNG data:
```bash
python main.py --generate-prng data/prng/generated.txt --num-samples=1000000 --seq-length=16
```
Generate TRNG data (fake/random for testing):
```bash
python main.py --generate-trng data/trng/generated.txt --num-samples=1000000 --seq-length=16
```

### 2. Prepare Binary Classification Data
Combine, label, shuffle, and split PRNG and TRNG data for binary classification:
```bash
python main.py --prepare-binary-data --prngfile=data/prng/generated.txt --trngfile=data/trng/generated.txt --seq-length=16
```
This creates:
- `data/binary/train.txt`, `data/binary/train_labels.txt`
- `data/binary/test.txt`, `data/binary/test_labels.txt`

### 3. Train the Model (Binary Classification)
Train on the combined and labeled data:
```bash
python main.py --train=prng --datafile=data/binary/train.txt --labelsfile=data/binary/train_labels.txt --epochs=10 --batch-size=1024 --seq-length=16
```

### 4. Test the Model
Test the trained model and print accuracy/confusion matrix:
```bash
python main.py --test=prng --model=pth/yourmodel.pth --datafile=data/binary/test.txt --labelsfile=data/binary/test_labels.txt --metrics --seq-length=16
```
Replace `yourmodel.pth` with the actual model filename saved during training.

### 5. Device Selection
The script will automatically use a GPU (CUDA) if available, otherwise it will use the CPU. The device in use is printed at startup.

---

## Additional CLI Options
- `--split-data input.txt train.txt test.txt` : Split a .txt file into train/test sets
- `--test-size 0.3` : Fraction of data to use for test set (default 0.3)
- `--epochs`, `--batch-size`, `--seq-length` : Control training parameters

---

## Notes
- All data files must contain one binary sequence per line (e.g., `0101010101010101`).
- For binary classification, always use the `--labelsfile` option to provide correct labels.
- For custom data, ensure PRNG and TRNG files have the same sequence length.

---

## Example Workflow
1. Generate PRNG and TRNG data
2. Prepare binary classification data
3. Train the model
4. Test the model and review metrics

---

For more details, see comments in `main.py` or run `python main.py` for help.

## Why Use a 1D CNN for TRNG/PRNG Analysis?
A 1D CNN is well-suited for detecting local patterns and dependencies in sequential data, such as bitstreams. By applying convolutional filters, the model can learn to recognize features that distinguish random from non-random sequences. This approach is more powerful than simple statistical tests, as it can learn subtle, complex patterns that may indicate non-randomness or bias in a generator.

## Features
- Generate or load binary sequence data (from .txt files)
- Train a CNN to classify sequences as PRNG or TRNG
- Save/load models with timestamped filenames
- Run inference on new data
- Fully configurable via CLI arguments
- Includes a comprehensive test suite

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd TRNG_CNN_Deep_Learning
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Format
- Data files must be plain text `.txt` files.
- Each line is a binary sequence (e.g., `0101010101010101`), all lines must be the same length (default: 16 bits).
- For PRNG data, files should be in `data/prng/`. For TRNG data, in `data/trng/`.

## Usage
### Generate PRNG Data
Generate a file with random PRNG sequences:
```bash
python main.py --generate-prng data/prng/generated.txt --num-samples=1000 --seq-length=16
```

### Train a Model
Train on PRNG or TRNG data from a `.txt` file:
```bash
python main.py --train=prng --datafile=data/prng/generated.txt
python main.py --train=trng --datafile=data/trng/yourfile.txt
```
- Models are saved to the `pth/` directory with a timestamped filename.

### Test a Model
Run inference on PRNG or TRNG data using a saved model:
```bash
python main.py --test=prng --model=pth/prng_YYYYMMDD_HHMMSS.pth --datafile=data/prng/generated.txt
python main.py --test=trng --model=pth/trng_YYYYMMDD_HHMMSS.pth --datafile=data/trng/yourfile.txt
```
- Predictions are printed to the console.

### Customization
- You can adjust `--epochs`, `--batch-size`, `--seq-length`, and `--num-samples` as needed.
- The model architecture automatically adapts to the sequence length.

## How It Works
- **Data Preparation:** Binary sequences are loaded from `.txt` files or generated randomly. Each sequence is treated as a 1D array of bits.
- **Model:** A 1D CNN processes each sequence, learning to classify it as PRNG or TRNG based on patterns in the bits.
- **Training:** The model is trained using cross-entropy loss to distinguish between the two classes.
- **Inference:** The trained model can be used to predict the class of new sequences.

## Running Tests
A comprehensive test suite is included. To run all tests:
```bash
pytest
```
This checks data loading, dataset, model, training, inference, and file generation.

## Contributing
Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

## License
MIT License. See `LICENSE` file for details.

## Tips
- Always match your `--seq-length` to the length of sequences in your data files.
- All model files are saved in the `pth/` directory (which is git-ignored).
- For real TRNG data, place your `.txt` files in `data/trng/`.
- The code is fully documented for easy understanding and extension.

---
For more details, see the docstrings in `main.py` or contact the project maintainer.
