# TRNG/PRNG CNN Quality Checker

A command-line tool for training and evaluating a 1D Convolutional Neural Network (CNN) on binary sequences, such as those produced by True Random Number Generators (TRNGs) or Pseudo-Random Number Generators (PRNGs).

## Step-by-Step Workflow

### 1. Prepare Your Data
- **Option A:** Generate PRNG data for testing:
  ```bash
  python main.py --generate-prng data/prng/generated.txt --num-samples=1000 --seq-length=16
  ```
- **Option B:** Collect or create your own `.txt` files for PRNG or TRNG sources. Each line should be a binary sequence (e.g., `0101010101010101`).

### 2. Split Data into Train/Test Sets
- Use the built-in splitter to create training and test sets:
  ```bash
  python main.py --split-data data/prng/generated.txt data/prng/train.txt data/prng/test.txt --test-size=0.3 --seq-length=16
  ```
- This will save two files: `train.txt` and `test.txt` with a 70/30 split by default.

### 3. Train the Model
- Train on your training set:
  ```bash
  python main.py --train=prng --datafile=data/prng/train.txt
  ```
- The trained model will be saved in the `pth/` directory with a timestamped filename.

### 4. Test the Model and Evaluate
- Run inference on your test set and print predictions:
  ```bash
  python main.py --test=prng --model=pth/prng_YYYYMMDD_HHMMSS.pth --datafile=data/prng/test.txt --metrics
  ```
- Add `--metrics` to print accuracy and a confusion matrix (assumes all test samples are of the same class).

### 5. Interpret Results
- **Accuracy**: Fraction of correct predictions (1.0 = perfect, 0.5 = random guessing for two classes).
- **Confusion Matrix**: Shows true/false positives/negatives. For PRNG-only test data, you expect all predictions to be class 0.

---

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
