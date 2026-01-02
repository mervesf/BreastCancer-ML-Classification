"""
Model Training Module for Breast Cancer Classification

Provides a unified interface for training and comparing multiple
machine learning algorithms with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field
import warnings
import json
from pathlib import Path

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib

warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """Container for model training results."""
    name: str
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    test_score: Optional[float] = None
    training_time: Optional[float] = None
    
    def __repr__(self) -> str:
        return f"ModelResult({self.name}: CV={self.cv_mean:.4f}+-{self.cv_std:.4f})"


class ModelTrainer:
    """
    Multi-algorithm model trainer with hyperparameter tuning.
    
    Supports training and comparing multiple ML algorithms:
    - Logistic Regression
    - Support Vector Machine (SVC)
    - K-Nearest Neighbors
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    
    Features:
    - GridSearchCV for hyperparameter optimization
    - Cross-validation for robust evaluation
    - Automatic model comparison
    - Model persistence (save/load)
    
    Example:
        >>> trainer = ModelTrainer()
        >>> trainer.add_model('logistic', LogisticRegression(), {...})
        >>> results = trainer.train_all(X_train, y_train)
        >>> best_model = trainer.get_best_model()
    """
    
    # Default hyperparameter grids
    DEFAULT_PARAM_GRIDS = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [500, 1000]
        },
        'svc': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        },
        'decision_tree': {
            'max_depth': [3, 5, 7, 10, None],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5],
            'bootstrap': [True, False]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }
    
    # Default models
    DEFAULT_MODELS = {
        'logistic_regression': LogisticRegression(random_state=42),
        'svc': SVC(random_state=42),
        'knn': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'accuracy',
                 random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the ModelTrainer.
        
        Args:
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for evaluation
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.models: Dict[str, Any] = {}
        self.param_grids: Dict[str, Dict] = {}
        self.results: Dict[str, ModelResult] = {}
        self.fitted_models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
    
    def add_model(self, name: str, model: Any, 
                  param_grid: Optional[Dict] = None) -> 'ModelTrainer':
        """
        Add a model to the trainer.
        
        Args:
            name: Name identifier for the model
            model: Sklearn model instance
            param_grid: Hyperparameter grid for GridSearchCV
            
        Returns:
            self for method chaining
        """
        self.models[name] = model
        if param_grid:
            self.param_grids[name] = param_grid
        return self
    
    def add_default_models(self, models: Optional[List[str]] = None) -> 'ModelTrainer':
        """
        Add default models with their parameter grids.
        
        Args:
            models: List of model names to add (None for all)
            
        Returns:
            self for method chaining
        """
        if models is None:
            models = list(self.DEFAULT_MODELS.keys())
        
        for name in models:
            if name in self.DEFAULT_MODELS:
                self.add_model(
                    name,
                    self.DEFAULT_MODELS[name],
                    self.DEFAULT_PARAM_GRIDS.get(name)
                )
        
        print(f"[OK] Added {len(models)} default models")
        return self
    
    def train_single(self, name: str, X_train: np.ndarray, 
                     y_train: np.ndarray, verbose: bool = True) -> ModelResult:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            name: Model name
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print progress
            
        Returns:
            ModelResult with training results
        """
        import time
        
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Add it first.")
        
        model = self.models[name]
        param_grid = self.param_grids.get(name, {})
        
        if verbose:
            print(f"Training {name}...", end=" ")
        
        start_time = time.time()
        
        # GridSearchCV for hyperparameter tuning
        if param_grid:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                random_state=self.random_state)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring=self.scoring,
                n_jobs=self.n_jobs, refit=True
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
        else:
            # No hyperparameter tuning
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
            best_score = cross_val_score(model, X_train, y_train, 
                                         cv=self.cv_folds).mean()
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                    cv=self.cv_folds, scoring=self.scoring)
        
        training_time = time.time() - start_time
        
        # Store results
        result = ModelResult(
            name=name,
            best_params=best_params,
            best_score=best_score,
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            training_time=training_time
        )
        
        self.results[name] = result
        self.fitted_models[name] = best_model
        
        if verbose:
            print(f"CV Score: {result.cv_mean:.4f} (+/- {result.cv_std:.4f}) "
                  f"[{training_time:.1f}s]")
        
        return result
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  verbose: bool = True) -> Dict[str, ModelResult]:
        """
        Train all added models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print progress
            
        Returns:
            Dictionary of ModelResults
        """
        if verbose:
            print("=" * 60)
            print("TRAINING ALL MODELS")
            print("=" * 60)
        
        for name in self.models:
            self.train_single(name, X_train, y_train, verbose)
        
        # Find best model
        self.best_model_name = max(self.results, key=lambda k: self.results[k].cv_mean)
        
        if verbose:
            print("=" * 60)
            print(f"Best Model: {self.best_model_name} "
                  f"(CV={self.results[self.best_model_name].cv_mean:.4f})")
            print("=" * 60)
        
        return self.results
    
    def evaluate_on_test(self, X_test: np.ndarray, 
                         y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with test scores
        """
        results_list = []
        
        for name, model in self.fitted_models.items():
            y_pred = model.predict(X_test)
            test_score = accuracy_score(y_test, y_pred)
            self.results[name].test_score = test_score
            
            results_list.append({
                'Model': name,
                'CV Score': f"{self.results[name].cv_mean:.4f}",
                'CV Std': f"{self.results[name].cv_std:.4f}",
                'Test Score': f"{test_score:.4f}",
                'Best Params': str(self.results[name].best_params)[:50] + "..."
            })
        
        return pd.DataFrame(results_list).sort_values('Test Score', ascending=False)
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Returns:
            (model_name, fitted_model)
        """
        if not self.best_model_name:
            raise ValueError("No models trained yet. Call train_all() first.")
        return self.best_model_name, self.fitted_models[self.best_model_name]
    
    def get_model(self, name: str) -> Any:
        """Get a specific fitted model by name."""
        if name not in self.fitted_models:
            raise ValueError(f"Model '{name}' not found or not trained.")
        return self.fitted_models[name]
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get a comparison DataFrame of all models.
        
        Returns:
            DataFrame comparing all model results
        """
        data = []
        for name, result in self.results.items():
            data.append({
                'Model': name,
                'CV Mean': result.cv_mean,
                'CV Std': result.cv_std,
                'Test Score': result.test_score or 'N/A',
                'Training Time (s)': f"{result.training_time:.2f}" if result.training_time else 'N/A'
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('CV Mean', ascending=False).reset_index(drop=True)
    
    def get_best_params_summary(self) -> pd.DataFrame:
        """Get best parameters for all models."""
        data = []
        for name, result in self.results.items():
            row = {'Model': name}
            row.update(result.best_params)
            data.append(row)
        return pd.DataFrame(data)
    
    # ==================== MODEL PERSISTENCE ====================
    
    def save_model(self, name: str, filepath: str) -> None:
        """
        Save a trained model to file.
        
        Args:
            name: Model name
            filepath: Path to save the model
        """
        if name not in self.fitted_models:
            raise ValueError(f"Model '{name}' not found or not trained.")
        
        joblib.dump(self.fitted_models[name], filepath)
        print(f"[OK] Model '{name}' saved to {filepath}")
    
    def save_best_model(self, filepath: str) -> None:
        """Save the best model to file."""
        if not self.best_model_name:
            raise ValueError("No best model found. Train models first.")
        self.save_model(self.best_model_name, filepath)
    
    def load_model(self, filepath: str, name: Optional[str] = None) -> Any:
        """
        Load a model from file.
        
        Args:
            filepath: Path to the saved model
            name: Optional name to register the model
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        if name:
            self.fitted_models[name] = model
        print(f"[OK] Model loaded from {filepath}")
        return model
    
    def save_results(self, filepath: str) -> None:
        """Save training results to JSON."""
        results_dict = {}
        for name, result in self.results.items():
            results_dict[name] = {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'cv_mean': result.cv_mean,
                'cv_std': result.cv_std,
                'test_score': result.test_score,
                'training_time': result.training_time
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"[OK] Results saved to {filepath}")
    
    def __repr__(self) -> str:
        trained = len(self.fitted_models)
        total = len(self.models)
        return f"ModelTrainer({trained}/{total} models trained)"
