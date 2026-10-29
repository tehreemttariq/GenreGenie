# GenreGenie - Source Code

## Member 1: Model Architecture & Training

### Files
- `model.py` - Model architectures (Matrix Factorization & Neural CF)
- `dataset.py` - Data loading and PyTorch datasets
- `train.py` - Training loop and hyperparameter tuning
- `utils.py` - Helper functions

### Models Implemented

#### 1. Matrix Factorization (MF)
- Simple collaborative filtering model
- Learns user and movie embeddings
- Uses biases and dropout for regularization

#### 2. Neural Collaborative Filtering (NCF)
- Advanced model combining embeddings with neural networks
- Multiple hidden layers (64 → 32)
- Better performance expected

### How to Run

#### Install Dependencies
```bash
pip install -r data_scripts/requirements.txt
```

#### Train Models
```bash
# From project root
python src/train.py
```

This will:
1. Load cleaned data from `data/cleaned/`
2. Split into train/val/test (70/15/15)
3. Train both MF and NCF models
4. Save best models to `models/`
5. Save training history to `results/`

#### Run Hyperparameter Search
Edit `train.py` and uncomment the last lines:
```python
hyperparameter_search()
```

### Hyperparameters Tested

| Parameter | Values |
|-----------|--------|
| Learning Rate | 0.001, 0.005 |
| Embedding Dim | 32, 50, 64 |
| Dropout | 0.1, 0.2, 0.3 |
| Epochs | 20, 30 |
| Batch Size | 256 |

### Outputs

**Models:**
- `models/matrix_factorization_best.pth`
- `models/neural_cf_best.pth`

**Results:**
- `results/training_history.json`
- `results/hyperparameter_results.csv`

### Training Time
- Matrix Factorization: ~2-3 minutes
- Neural CF: ~3-5 minutes
(On CPU, faster on GPU)
