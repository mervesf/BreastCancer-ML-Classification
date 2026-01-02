"""
Breast Cancer ML Classification

A comprehensive machine learning toolkit for breast cancer diagnosis
using multiple classification algorithms.
"""

from .data_processor import DataProcessor
from .model_trainer import ModelTrainer, ModelResult
from .evaluator import ModelEvaluator, EvaluationResult, quick_evaluate
from .visualizer import Visualizer

__version__ = '1.0.0'
__author__ = 'Merve'
__all__ = [
    'DataProcessor',
    'ModelTrainer',
    'ModelResult',
    'ModelEvaluator',
    'EvaluationResult',
    'Visualizer',
    'quick_evaluate'
]
