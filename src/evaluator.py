"""
Model Evaluation Module for Breast Cancer Classification

Provides comprehensive model evaluation tools including:
- Classification metrics
- Confusion matrix analysis
- ROC curves and AUC
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    confusion_matrix: np.ndarray
    classification_report: str
    
    def __repr__(self) -> str:
        return (f"EvaluationResult(Accuracy={self.accuracy:.4f}, "
                f"Precision={self.precision:.4f}, Recall={self.recall:.4f}, "
                f"F1={self.f1:.4f})")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'Accuracy': self.accuracy,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1 Score': self.f1,
            'Specificity': self.specificity
        }


class ModelEvaluator:
    """
    Comprehensive model evaluator for classification tasks.
    
    Provides detailed evaluation metrics and analysis for
    breast cancer classification models.
    
    Features:
    - Standard classification metrics
    - Confusion matrix analysis
    - ROC curve and AUC calculation
    - Feature importance extraction
    - Multi-model comparison
    
    Example:
        >>> evaluator = ModelEvaluator()
        >>> result = evaluator.evaluate(model, X_test, y_test)
        >>> print(result.accuracy)
        >>> evaluator.plot_confusion_matrix(result)
    """
    
    # Class labels for breast cancer
    CLASS_NAMES = ['Benign (0)', 'Malignant (1)']
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            class_names: Optional custom class names
        """
        self.class_names = class_names or self.CLASS_NAMES
        self.results: Dict[str, EvaluationResult] = {}
    
    def evaluate(self, model: Any, X_test: np.ndarray, 
                 y_test: np.ndarray, model_name: str = 'model') -> EvaluationResult:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained sklearn model
            X_test: Test features
            y_test: True labels
            model_name: Name identifier for the model
            
        Returns:
            EvaluationResult with all metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                       target_names=self.class_names)
        
        result = EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            specificity=specificity,
            confusion_matrix=cm,
            classification_report=report
        )
        
        self.results[model_name] = result
        return result
    
    def evaluate_multiple(self, models: Dict[str, Any], 
                          X_test: np.ndarray, 
                          y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate multiple models and compare.
        
        Args:
            models: Dictionary of {name: model}
            X_test: Test features
            y_test: True labels
            
        Returns:
            DataFrame comparing all models
        """
        comparison = []
        
        for name, model in models.items():
            result = self.evaluate(model, X_test, y_test, name)
            row = {'Model': name}
            row.update(result.to_dict())
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        return df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
    
    def get_confusion_matrix_stats(self, result: EvaluationResult) -> Dict[str, int]:
        """
        Get detailed confusion matrix statistics.
        
        Args:
            result: EvaluationResult object
            
        Returns:
            Dictionary with TP, TN, FP, FN counts
        """
        tn, fp, fn, tp = result.confusion_matrix.ravel()
        return {
            'True Negatives (TN)': tn,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn,
            'True Positives (TP)': tp,
            'Total': tn + fp + fn + tp
        }
    
    def get_roc_data(self, model: Any, X_test: np.ndarray, 
                     y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate ROC curve data.
        
        Args:
            model: Trained model with predict_proba
            X_test: Test features
            y_test: True labels
            
        Returns:
            (fpr, tpr, auc_score)
        """
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            raise ValueError("Model doesn't support probability predictions")
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, auc_score
    
    def get_precision_recall_data(self, model: Any, X_test: np.ndarray,
                                   y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate Precision-Recall curve data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            
        Returns:
            (precision, recall, average_precision)
        """
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            raise ValueError("Model doesn't support probability predictions")
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        return precision, recall, avg_precision
    
    def get_feature_importance(self, model: Any, 
                               feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances sorted
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model doesn't have feature_importances_ or coef_")
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        return df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    def print_summary(self, model_name: str) -> None:
        """
        Print evaluation summary for a model.
        
        Args:
            model_name: Name of the evaluated model
        """
        if model_name not in self.results:
            raise ValueError(f"No results for model '{model_name}'")
        
        result = self.results[model_name]
        
        print("=" * 60)
        print(f"EVALUATION SUMMARY: {model_name}")
        print("=" * 60)
        print(f"\nAccuracy:    {result.accuracy:.4f}")
        print(f"Precision:   {result.precision:.4f}")
        print(f"Recall:      {result.recall:.4f}")
        print(f"F1 Score:    {result.f1:.4f}")
        print(f"Specificity: {result.specificity:.4f}")
        print("\nConfusion Matrix:")
        print(result.confusion_matrix)
        print("\nClassification Report:")
        print(result.classification_report)
        print("=" * 60)
    
    def get_medical_interpretation(self, result: EvaluationResult) -> Dict[str, str]:
        """
        Get medical interpretation of results.
        
        For breast cancer:
        - False Negative = Missed cancer (dangerous!)
        - False Positive = Unnecessary further testing
        
        Args:
            result: EvaluationResult object
            
        Returns:
            Dictionary with medical interpretations
        """
        stats = self.get_confusion_matrix_stats(result)
        total = stats['Total']
        
        return {
            'Sensitivity (Recall)': (
                f"{result.recall:.1%} of actual cancer cases were correctly identified. "
                f"({stats['True Positives (TP)']} out of {stats['True Positives (TP)'] + stats['False Negatives (FN)']})"
            ),
            'Specificity': (
                f"{result.specificity:.1%} of benign cases were correctly identified. "
                f"({stats['True Negatives (TN)']} out of {stats['True Negatives (TN)'] + stats['False Positives (FP)']})"
            ),
            'Missed Cancers (FN)': (
                f"{stats['False Negatives (FN)']} cancer cases were missed. "
                f"This is critical as it could delay treatment."
            ),
            'False Alarms (FP)': (
                f"{stats['False Positives (FP)']} benign cases were flagged as cancer. "
                f"This leads to unnecessary anxiety and additional testing."
            ),
            'Overall': (
                f"The model correctly classified {result.accuracy:.1%} of all cases "
                f"({stats['True Positives (TP)'] + stats['True Negatives (TN)']} out of {total})."
            )
        }


def quick_evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Quick evaluation function returning basic metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary with basic metrics
    """
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary')
    }
