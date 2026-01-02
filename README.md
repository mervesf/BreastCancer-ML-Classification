<p align="center">
  <img src="assets/banner.png" alt="Breast Cancer ML Classification" width="800"/>
</p>

<h1 align="center">🎗️ Breast Cancer ML Classification</h1>

<p align="center">
  <strong>Multi-Algorithm Machine Learning for Breast Cancer Diagnosis</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#algorithms">Algorithms</a> •
  <a href="#results">Results</a> •
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.0+-orange.svg" alt="Sklearn"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/ML-Classification-red.svg" alt="ML"/>
  <img src="https://img.shields.io/badge/Healthcare-AI-purple.svg" alt="Healthcare"/>
</p>

---

## 📊 Overview

Compare **6 machine learning algorithms** for breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer dataset. The project provides automated hyperparameter tuning, comprehensive evaluation metrics, and medical interpretation of results.

<p align="center">
  <img src="assets/dashboard.png" alt="Results Dashboard" width="800"/>
</p>

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **6 ML Algorithms** | Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting |
| 🎯 **GridSearchCV** | Automated hyperparameter optimization |
| 📊 **Cross-Validation** | Robust 5-fold stratified CV |
| 🔍 **Outlier Detection** | IQR-based outlier handling |
| 📈 **Feature Selection** | Correlation-based feature reduction |
| 📉 **Comprehensive Metrics** | Accuracy, Precision, Recall, F1, Specificity, ROC-AUC |
| 🏥 **Medical Interpretation** | Clinical context for False Negatives/Positives |
| 📊 **Visualizations** | Confusion matrix, ROC curves, feature importance |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw Data ──► DataProcessor ──► ModelTrainer ──► Evaluator      │
│                    │                  │              │          │
│              - Clean data      - 6 Algorithms   - Metrics       │
│              - Handle outliers - GridSearchCV   - Confusion     │
│              - Scale features  - Cross-Val      - ROC Curve     │
│              - Feature select  - Compare        - Interpret     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/BreastCancer-ML-Classification.git
cd BreastCancer-ML-Classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📦 Project Structure

```
BreastCancer-ML-Classification/
├── 📂 src/
│   ├── data_processor.py   # Data loading, cleaning, preprocessing
│   ├── model_trainer.py    # Multi-model training with GridSearchCV
│   ├── evaluator.py        # Metrics, confusion matrix, ROC
│   └── visualizer.py       # All visualizations
├── 📂 notebooks/
│   └── demo.ipynb          # Interactive demonstration
├── 📂 data/
│   └── breast_cancer.csv   # Dataset (or load from sklearn)
├── 📂 models/
│   └── (saved models)
├── 📂 output/
│   └── (generated plots)
├── 📂 tests/
│   └── test_pipeline.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 🎯 Quick Start

### Basic Usage

```python
from src import DataProcessor, ModelTrainer, ModelEvaluator, Visualizer

# 1. Load and preprocess data
processor = DataProcessor()
processor.load_from_sklearn()  # or processor.load_data('data.csv')
processor.clean_data()
processor.handle_outliers(method='clip')
processor.scale_features(method='minmax')

# 2. Prepare train/test split
X_train, X_test, y_train, y_test = processor.prepare_for_training(test_size=0.2)

# 3. Train multiple models
trainer = ModelTrainer(cv_folds=5)
trainer.add_default_models()  # Adds all 6 algorithms
results = trainer.train_all(X_train, y_train)

# 4. Evaluate on test data
comparison = trainer.evaluate_on_test(X_test, y_test)
print(comparison)

# 5. Get best model
best_name, best_model = trainer.get_best_model()
print(f"Best Model: {best_name}")

# 6. Detailed evaluation
evaluator = ModelEvaluator()
result = evaluator.evaluate(best_model, X_test, y_test, best_name)
evaluator.print_summary(best_name)
```

### Visualize Results

```python
viz = Visualizer()

# Confusion Matrix
viz.plot_confusion_matrix(result.confusion_matrix, save_path='output/cm.png')

# Model Comparison
viz.plot_model_comparison(comparison, save_path='output/comparison.png')

# Feature Importance (for tree-based models)
importance = evaluator.get_feature_importance(best_model, processor.get_feature_names())
viz.plot_feature_importance(importance, save_path='output/importance.png')

# Full Dashboard
viz.create_dashboard(result.confusion_matrix, comparison, importance)
```

## 🤖 Algorithms

| Algorithm | Type | Hyperparameters Tuned |
|-----------|------|----------------------|
| **Logistic Regression** | Linear | C, penalty, solver, max_iter |
| **SVM (SVC)** | Kernel-based | C, kernel, gamma, degree |
| **K-Nearest Neighbors** | Instance-based | n_neighbors, weights, metric |
| **Decision Tree** | Tree-based | max_depth, criterion, min_samples |
| **Random Forest** | Ensemble | n_estimators, max_depth, bootstrap |
| **Gradient Boosting** | Ensemble | n_estimators, learning_rate, max_depth |

## 📊 Results

### Model Comparison (Sample Results)

| Model | CV Score | Test Accuracy | Training Time |
|-------|----------|---------------|---------------|
| Logistic Regression | 0.9736 | 0.9649 | 2.3s |
| Random Forest | 0.9648 | 0.9561 | 15.7s |
| SVM (RBF) | 0.9692 | 0.9561 | 8.4s |
| Gradient Boosting | 0.9604 | 0.9474 | 12.1s |
| KNN | 0.9560 | 0.9386 | 5.2s |
| Decision Tree | 0.9297 | 0.9211 | 1.8s |

### Confusion Matrix Analysis

```
              Predicted
              Benign  Malignant
Actual Benign    70        2
     Malignant    2       40

Accuracy:    96.49%
Sensitivity: 95.24% (Cancer detection rate)
Specificity: 97.22% (Benign detection rate)
```

### Medical Interpretation

- **False Negatives (2)**: Missed cancer cases - requires clinical attention
- **False Positives (2)**: Unnecessary further testing
- **High Sensitivity**: Critical for cancer screening

## 📖 Documentation

### DataProcessor Methods

| Method | Description |
|--------|-------------|
| `load_data(path)` | Load from CSV |
| `load_from_sklearn()` | Load from sklearn datasets |
| `clean_data()` | Remove nulls, encode labels |
| `detect_outliers()` | Find outliers using IQR |
| `handle_outliers(method)` | Clip, remove, or replace |
| `scale_features(method)` | MinMax or Standard scaling |
| `remove_correlated_features()` | Remove highly correlated |
| `prepare_for_training()` | Get train/test split |

### ModelTrainer Methods

| Method | Description |
|--------|-------------|
| `add_model(name, model, params)` | Add custom model |
| `add_default_models()` | Add all 6 algorithms |
| `train_all(X, y)` | Train with GridSearchCV |
| `get_best_model()` | Get best performing model |
| `evaluate_on_test(X, y)` | Evaluate all on test set |
| `save_best_model(path)` | Save to file |

### Evaluation Metrics

| Metric | Formula | Importance |
|--------|---------|------------|
| **Accuracy** | (TP+TN)/Total | Overall correctness |
| **Precision** | TP/(TP+FP) | Positive predictive value |
| **Recall** | TP/(TP+FN) | Sensitivity, cancer detection |
| **Specificity** | TN/(TN+FP) | True negative rate |
| **F1 Score** | 2×(P×R)/(P+R) | Balance of P and R |

## 🧪 Running Tests

```bash
cd tests
python test_pipeline.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [UCI ML Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [GridSearchCV Guide](https://scikit-learn.org/stable/modules/grid_search.html)

---

<p align="center">
  Built by <a href="https://github.com/mervesf">Merve</a> · ⭐ Star if you found this useful!
</p>

<p align="center">
  <i>Early detection saves lives 🎗️</i>
</p>
