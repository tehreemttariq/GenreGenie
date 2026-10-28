
## 📋 **Work Division for Phase 3 & 4**

### **Member 1 (You) - Model Architecture & Core Training** 🔵

**Files to create:**
1. `src/model.py` - Model architectures (Matrix Factorization + Neural CF)
2. `src/dataset.py` - Data loading and preprocessing for PyTorch
3. `src/train.py` - Main training loop with hyperparameter tuning
4. `src/utils.py` - Helper functions (save/load models, set seed)

**Responsibilities:**
- Implement 2 models: Matrix Factorization (simple) + Neural Collaborative Filtering (advanced)
- Create PyTorch datasets and dataloaders
- Build training loop with loss tracking
- Hyperparameter tuning (learning rate, batch size, embedding dim, dropout)
- Save model checkpoints to `models/` folder

**Key Functions:**
- `MatrixFactorization` class
- `NeuralCollaborativeFiltering` class
- `train_epoch()` function
- `hyperparameter_search()` function

---

### **Member 2 (Partner) - Evaluation & Analysis** 🟢

**Files to create:**
1. `src/evaluate.py` - Evaluation metrics and testing
2. `src/visualize.py` - Generate plots (loss curves, confusion matrix, etc.)
3. `src/inference.py` - Make predictions and recommendations
4. `src/bias_analysis.py` - Bias check and error analysis

**Responsibilities:**
- Implement evaluation metrics (RMSE, MAE, Precision@K, Recall@K, NDCG)
- Create train/validation/test split (70/15/15)
- Generate visualization plots (loss/accuracy curves, metric comparisons)
- Bias analysis: check model performance across user groups
- Error analysis: identify where model fails

**Key Functions:**
- `evaluate_model()` function
- `calculate_metrics()` function
- `plot_training_curves()` function
- `recommend_movies()` function
- `analyze_bias()` function

---

## 📊 **Shared Components**

**Both members collaborate on:**
1. `requirements.txt` - Add PyTorch dependencies
2. `README.md` - Update with Phase 3/4 instructions
3. `docs/report.tex` - Each member documents their part
4. Git commits - Each member commits their own code

---

## 🔄 **Integration Points**

**How it connects:**
1. Member 1 trains model → saves to `models/best_model.pth`
2. Member 2 loads model → runs evaluation → generates plots
3. Both share results through `results/` folder:
   - `results/training_history.json` (Member 1)
   - `results/evaluation_metrics.json` (Member 2)
   - `results/figures/` (Member 2)

---

## 📝 **Detailed Task Breakdown**

### **Member 1 Tasks (Model & Training):**

**Week 1:**
- [ ] Create `src/model.py` with MatrixFactorization class
- [ ] Create `src/dataset.py` for data loading
- [ ] Create `src/utils.py` with helper functions

**Week 2:**
- [ ] Create `src/train.py` with training loop
- [ ] Add NeuralCollaborativeFiltering to `model.py`
- [ ] Run hyperparameter experiments
- [ ] Save best models and training history

**Deliverables:**
- 2 trained models in `models/` folder
- `results/hyperparameter_results.csv` table
- Git commits showing implementation progress

---

### **Member 2 Tasks (Evaluation & Analysis):**

**Week 1:**
- [ ] Create `src/evaluate.py` with metrics functions
- [ ] Create train/val/test split function
- [ ] Create `src/visualize.py` for plotting

**Week 2:**
- [ ] Create `src/inference.py` for recommendations
- [ ] Create `src/bias_analysis.py` for fairness check
- [ ] Generate all evaluation plots
- [ ] Write error analysis section

**Deliverables:**
- `results/evaluation_metrics.json` with all metrics
- `results/figures/` with 6+ plots
- `results/bias_report.txt` with analysis
- Git commits showing evaluation work

---

## 📦 **File Structure (Final)**

```
GenreGenie/
├── src/
│   ├── model.py              (Member 1)
│   ├── dataset.py            (Member 1)
│   ├── train.py              (Member 1)
│   ├── utils.py              (Member 1)
│   ├── evaluate.py           (Member 2)
│   ├── visualize.py          (Member 2)
│   ├── inference.py          (Member 2)
│   └── bias_analysis.py      (Member 2)
├── models/
│   ├── matrix_factorization_best.pth
│   └── neural_cf_best.pth
├── results/
│   ├── training_history.json
│   ├── evaluation_metrics.json
│   ├── hyperparameter_results.csv
│   ├── bias_report.txt
│   └── figures/
│       ├── training_curves.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── precision_recall_curve.png
│       ├── rating_distribution_pred_vs_true.png
│       └── bias_analysis.png
└── requirements.txt
```

---

## ⚙️ **Technical Specifications**

### **Models to Implement:**

1. **Matrix Factorization (Simpler - Good baseline)**
   - User embeddings + Movie embeddings
   - Biases
   - Dropout regularization

2. **Neural Collaborative Filtering (Advanced)**
   - Combines MF with neural networks
   - Multiple hidden layers
   - Better performance expected

### **Hyperparameters to Tune:**

| Parameter | Values to Test | Who Tunes |
|-----------|----------------|-----------|
| Learning Rate | [0.001, 0.005, 0.01] | Member 1 |
| Batch Size | [128, 256, 512] | Member 1 |
| Embedding Dim | [32, 50, 64] | Member 1 |
| Dropout | [0.1, 0.2, 0.3] | Member 1 |
| Epochs | [20, 30, 50] | Member 1 |

### **Metrics to Report:**

| Metric | Purpose | Who Implements |
|--------|---------|----------------|
| RMSE | Rating prediction error | Member 2 |
| MAE | Average absolute error | Member 2 |
| Precision@K | Top-K recommendation quality | Member 2 |
| Recall@K | Coverage of relevant items | Member 2 |
| NDCG@K | Ranking quality | Member 2 |

---

## 💡 **Why This Division Works:**

✅ **Clear separation:** Model building vs. Evaluation  
✅ **Equal complexity:** Both require similar effort  
✅ **Independent work:** Can work in parallel  
✅ **Easy integration:** Simple file handoff (saved models)  
✅ **Git friendly:** Different files = no merge conflicts  
✅ **Fair credit:** Each member's work visible in commits  

---

**Which role do you want? Member 1 (Models & Training) or Member 2 (Evaluation & Analysis)?**

Once you decide, I'll create all the files and detailed implementation guide for your part! 🚀