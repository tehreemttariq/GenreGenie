# GenreGenie - Source Code

### Files
- `model.py` - Model architectures (Matrix Factorization & Neural CF)
- `dataset.py` - Data loading and PyTorch datasets
- `train.py` - Training loop and hyperparameter tuning
- `evaluate.py` - Rating and top-K recommendation metrics
- `inference.py` - Load a trained model and recommend unseen movies
- `bias_analysis.py` - User-activity, genre, and popularity analysis
- `visualize.py` - Evaluation plots
- `run_evaluation.py` - Reproduce metrics, plots, and analysis artifacts
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
3. Train SVD and baseline collaborative-filtering models
4. Save the models to `models/`
5. Save training and hyperparameter results to `results/`

#### Run Evaluation
From the project root:
```bash
python src/run_evaluation.py
```

This evaluates the saved SVD and baseline models, selects the model with the
lowest validation RMSE for recommendation analysis, and writes metrics,
recommendations, bias analysis, and eight plots to `results/`.

#### Get Recommendations
From the project root:
```bash
python src/inference.py --user-id 1 --model baseline --top-n 10
```

Supported model names are `baseline` and `svd`. Recommendations exclude movies
already rated in the training split.

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

**Models used by the current training script:**
- `models/svd_model.pkl`
- `models/baseline_model.pkl`

**Results:**
- `results/training_results.json`
- `results/hyperparameter_results.csv`
- `results/evaluation_metrics.json`
- `results/bias_report.txt`
- `results/figures/`

### Training Time
- Matrix Factorization: ~2-3 minutes
- Neural CF: ~3-5 minutes
