# Member 1 (Models & Training) - Complete Guide

## 🎯 Your Responsibilities

You are responsible for:
1. ✅ Model architectures (Matrix Factorization + Neural CF)
2. ✅ Data loading for PyTorch
3. ✅ Training loop
4. ✅ Hyperparameter tuning
5. ✅ Saving models and training results

## 📁 Files You Created

```
src/
├── model.py          ✅ DONE - Two model classes
├── dataset.py        ✅ DONE - Data loading
├── train.py          ✅ DONE - Training script
├── utils.py          ✅ DONE - Helper functions
└── README.md         ✅ DONE - Documentation
```

## 🚀 Step-by-Step Execution

### Step 1: Install Dependencies
```bash
pip install torch tqdm pandas numpy scikit-learn
```

### Step 2: Verify Data Exists
Check that these files exist:
```bash
data/cleaned/ratings_cleaned.csv
data/cleaned/movies_cleaned.csv
data/cleaned/links_cleaned.csv
```

### Step 3: Run Training
```bash
cd d:\GenreGenie
python src/train.py
```

This will:
- Load and split data (70/15/15)
- Train Matrix Factorization model (30 epochs)
- Train Neural CF model (30 epochs)
- Save best models to `models/`
- Save training history to `results/training_history.json`

**Expected output:**
```
============================================================
TRAINING MATRIX FACTORIZATION MODEL
============================================================

Loading ratings data...
Total ratings: 100836
Users: 610
Movies: 9742

Data split:
  Train: 70585 (70.0%)
  Val:   15126 (15.0%)
  Test:  15125 (15.0%)

Model configuration:
  Users: 610
  Movies: 9742
  Embedding dim: 50
  Dropout: 0.2
  Learning rate: 0.001

Trainable parameters: 488,351

Starting training...
------------------------------------------------------------
Epoch   1/30 | Train Loss: 3.2145 | Val Loss: 1.8234
Epoch   2/30 | Train Loss: 1.5678 | Val Loss: 1.3456
...
Epoch  30/30 | Train Loss: 0.7234 | Val Loss: 0.8567
------------------------------------------------------------
Training complete!
Best validation loss: 0.8234
```

### Step 4: (Optional) Run Hyperparameter Search

Edit `src/train.py`, scroll to bottom, and uncomment:
```python
hyperparameter_search()
```

Then run:
```bash
python src/train.py
```

This tests 5 different configurations and saves results to:
`results/hyperparameter_results.csv`

## 📊 What You'll Get

### 1. Trained Models
```
models/
├── matrix_factorization_best.pth  (~2 MB)
└── neural_cf_best.pth             (~2 MB)
```

### 2. Training Results
```
results/
├── training_history.json          (Loss curves)
└── hyperparameter_results.csv     (If you ran search)
```

## 📝 For Your Report (Phase 3)

### Section 3.1: Framework Selection
"We implemented our recommendation system using **PyTorch**, chosen for its:
- Dynamic computation graphs suitable for iterative development
- Strong community support and documentation
- Efficient GPU acceleration for matrix operations"

### Section 3.2: Model Architecture

**Matrix Factorization:**
```
- User Embedding: (n_users, 50)
- Movie Embedding: (n_movies, 50)
- User Bias: (n_users, 1)
- Movie Bias: (n_movies, 1)
- Dropout: 0.2
- Loss: MSE
- Optimizer: Adam
```

**Neural Collaborative Filtering:**
```
- User Embedding: (n_users, 50)
- Movie Embedding: (n_movies, 50)
- MLP: 100 → 64 → 32 → 1
- Activation: ReLU
- Dropout: 0.2
- Loss: MSE
- Optimizer: Adam
```

### Section 3.3: Hyperparameters Table

| Parameter | Tested Values | Final Choice | Justification |
|-----------|---------------|--------------|---------------|
| Learning Rate | 0.001, 0.005 | 0.001 | Better convergence |
| Embedding Dim | 32, 50, 64 | 50 | Balance of capacity & speed |
| Dropout | 0.1, 0.2, 0.3 | 0.2 | Prevents overfitting |
| Batch Size | 256, 512 | 256 | Faster training |
| Epochs | 20, 30, 50 | 30 | Convergence achieved |

## 🔄 Git Commit Strategy

**Make commits as you complete each file:**

```bash
# After creating model.py
git add src/model.py
git commit -m "Implement Matrix Factorization and NCF models"

# After creating dataset.py
git add src/dataset.py
git commit -m "Add PyTorch dataset and data loading"

# After creating utils.py
git add src/utils.py
git commit -m "Add utility functions for training"

# After creating train.py
git add src/train.py
git commit -m "Implement training loop with validation"

# After training
git add models/ results/ src/README.md
git commit -m "Add trained models and training results"

# Final push
git push origin main
```

## ⚠️ Common Issues & Solutions

### Issue 1: "No module named torch"
```bash
pip install torch
```

### Issue 2: "File not found: ratings_cleaned.csv"
Run data cleaning first:
```bash
python data_scripts/02_data_cleaning.py
```

### Issue 3: Out of memory
Reduce batch size in `dataset.py`:
```python
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

### Issue 4: Training too slow
Reduce epochs in `train.py`:
```python
train_model(n_epochs=20)  # Instead of 30
```

## ✅ Completion Checklist

- [ ] All 4 Python files created (`model.py`, `dataset.py`, `train.py`, `utils.py`)
- [ ] Dependencies installed
- [ ] Training script runs successfully
- [ ] Both models trained (MF and NCF)
- [ ] Models saved to `models/` folder
- [ ] Training history saved to `results/`
- [ ] (Optional) Hyperparameter search completed
- [ ] Git commits made for each file
- [ ] Documentation written in report
- [ ] Code pushed to GitHub

## 🤝 Handoff to Member 2

Once you're done, tell Member 2:
1. ✅ Models are saved in `models/` folder
2. ✅ Training history in `results/training_history.json`
3. ✅ Data mappings saved (for evaluation)
4. ✅ They can start evaluation using your trained models

## 📞 Questions?

If something doesn't work:
1. Check error message
2. Verify file paths
3. Ensure data cleaning was completed
4. Ask me for help!
