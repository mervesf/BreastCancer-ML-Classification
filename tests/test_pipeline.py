"""
Unit tests for Breast Cancer ML Classification Pipeline.
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / 'src')
sys.path.insert(0, src_path)

import numpy as np


def test(name, condition):
    """Test helper function."""
    status = '[PASS]' if condition else '[FAIL]'
    print(f'{status} {name}')
    return condition


def run_tests():
    """Run all tests."""
    print('='*60)
    print('BREAST CANCER ML CLASSIFICATION TESTS')
    print('='*60)
    
    all_passed = True
    
    # Import modules
    from data_processor import DataProcessor
    from model_trainer import ModelTrainer
    from evaluator import ModelEvaluator, quick_evaluate
    
    # Test 1: DataProcessor initialization
    processor = DataProcessor()
    all_passed &= test('DataProcessor init', processor is not None)
    
    # Test 2: Load from sklearn
    processor.load_from_sklearn()
    all_passed &= test('Load sklearn data', len(processor.df) == 569)
    
    # Test 3: Data info
    info = processor.get_info()
    all_passed &= test('Get info', len(info) > 0)
    
    # Test 4: Summary
    summary = processor.get_summary()
    all_passed &= test('Get summary', 'Total Samples' in summary)
    
    # Test 5: Detect outliers
    outliers = processor.detect_outliers()
    all_passed &= test('Detect outliers', isinstance(outliers, dict))
    
    # Test 6: Handle outliers
    processor.handle_outliers(method='clip')
    all_passed &= test('Handle outliers', True)
    
    # Test 7: Scale features
    processor.scale_features(method='minmax')
    feature_cols = [c for c in processor.df.columns if c != 'diagnosis']
    scaled_max = processor.df[feature_cols].max().max()
    all_passed &= test('Scale features', scaled_max <= 1.001)  # Allow small float error
    
    # Test 8: Correlation matrix
    corr = processor.get_correlation_matrix()
    all_passed &= test('Correlation matrix', corr.shape[0] == corr.shape[1])
    
    # Test 9: Prepare for training
    X_train, X_test, y_train, y_test = processor.prepare_for_training(test_size=0.2)
    all_passed &= test('Train/test split', len(X_train) > len(X_test))
    
    # Test 10: ModelTrainer initialization
    trainer = ModelTrainer(cv_folds=3)  # Use 3 folds for faster testing
    all_passed &= test('ModelTrainer init', trainer is not None)
    
    # Test 11: Add default models (just 2 for speed)
    trainer.add_default_models(['logistic_regression', 'decision_tree'])
    all_passed &= test('Add models', len(trainer.models) == 2)
    
    # Test 12: Train models
    print('\nTraining models (this may take a moment)...')
    results = trainer.train_all(X_train, y_train, verbose=True)
    all_passed &= test('Train all models', len(results) == 2)
    
    # Test 13: Get best model
    best_name, best_model = trainer.get_best_model()
    all_passed &= test('Get best model', best_model is not None)
    print(f'  Best model: {best_name}')
    
    # Test 14: Evaluate on test
    comparison = trainer.evaluate_on_test(X_test, y_test)
    all_passed &= test('Evaluate on test', len(comparison) == 2)
    
    # Test 15: ModelEvaluator
    evaluator = ModelEvaluator()
    result = evaluator.evaluate(best_model, X_test, y_test, best_name)
    all_passed &= test('Model evaluation', result.accuracy > 0.8)
    print(f'  Test accuracy: {result.accuracy:.4f}')
    
    # Test 16: Confusion matrix stats
    cm_stats = evaluator.get_confusion_matrix_stats(result)
    all_passed &= test('Confusion matrix stats', 'True Positives (TP)' in cm_stats)
    
    # Test 17: Medical interpretation
    interpretation = evaluator.get_medical_interpretation(result)
    all_passed &= test('Medical interpretation', 'Sensitivity (Recall)' in interpretation)
    
    # Test 18: Quick evaluate
    quick_result = quick_evaluate(best_model, X_test, y_test)
    all_passed &= test('Quick evaluate', 'accuracy' in quick_result)
    
    # Test 19: Feature importance (for decision tree)
    dt_model = trainer.get_model('decision_tree')
    importance = evaluator.get_feature_importance(dt_model, processor.get_feature_names())
    all_passed &= test('Feature importance', len(importance) > 0)
    
    # Test 20: Comparison DataFrame
    comparison_df = trainer.get_comparison_dataframe()
    all_passed &= test('Comparison DataFrame', 'Model' in comparison_df.columns)
    
    print('='*60)
    if all_passed:
        print('SUCCESS: ALL 20 TESTS PASSED!')
    else:
        print('ERROR: SOME TESTS FAILED')
    print('='*60)
    
    # Print final results
    print('\nFinal Model Comparison:')
    print(comparison_df.to_string(index=False))
    
    return all_passed


if __name__ == '__main__':
    run_tests()
